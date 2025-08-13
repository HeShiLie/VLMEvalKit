from typing import List, Dict, Optional
import re
import io
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from .video_tools_gao_v1 import VideoToolBox
from .visual_toolbox_v5 import VisualToolBoxV5
from .video_toolbox_gaozhe_v5 import VideoToolBoxV5
from .parallel_envs import ParallelEnv

import warnings

def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)

def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}")

def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor

def _concat_vllm_input(prompt_token_ids, response_token_ids, tokenizer=None):
    # NOTE: temporarily fix qwen-base oov issue
    if tokenizer is not None:
        max_token_id = max(tokenizer.get_vocab().values())
        tokenizer_size = len(tokenizer)
        max_token_id = max(max_token_id, tokenizer_size)
        valid_token_mask = torch.le(response_token_ids, max_token_id)
        response_token_ids = torch.masked_select(response_token_ids, valid_token_mask)

    if isinstance(prompt_token_ids, torch.Tensor):
        output_tensor = torch.cat([
            prompt_token_ids,
            response_token_ids.to(prompt_token_ids.device),
        ], dim=-1)
        return output_tensor.cpu().numpy().flatten().tolist()
    else:
        output_array = np.concatenate([
            prompt_token_ids,
            response_token_ids.cpu().numpy(),
        ], axis=-1)
        return output_array.flatten().tolist()

def _merge_multi_modal_inputs(mm_input, other):
    if not mm_input and not other:
        return {}
    elif len(mm_input) == 0 and len(other) > 0:
        return other
    elif len(mm_input) > 0 and len(other) == 0:
        return mm_input

    output_dict = {}
    for key in mm_input.keys():
        if key not in other.keys():
            output_dict[key] = mm_input[key]
            continue

        mm_value = mm_input[key]
        other_value = other.pop(key)
        if isinstance(mm_value, np.ndarray) and isinstance(other_value, np.ndarray):
            merged_value = np.concatenate([mm_value, other_value], axis=0)
        elif isinstance(mm_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            merged_value = torch.cat([mm_value, other_value], dim=0)
        else:
            raise ValueError(f"Invalid {type(mm_value)=}, {type(other_value)=}")

        output_dict[key] = merged_value
    return dict(**output_dict, **other)

def _pad_tool_lists(input_list: List[List[str]], pad_token: str = "", max_len: int = None) -> List[List[str]]:
    if max_len is None:
        max_len = max(len(sublist) for sublist in input_list)
    return [sublist + [pad_token] * (max_len - len(sublist)) for sublist in input_list]

def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    """
    pad a 2D list to a 2D tensor.
    """
    if not response:
        print("[DEBUG] Warning: Empty response list in pad_2d_list_to_length")
        return torch.empty(0, 0, dtype=torch.long)
    
    response_length = max(len(sub_list) for sub_list in response)
    target_length = max_length if max_length is not None and max_length > response_length else response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor

def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1546
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0

                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids

def agent_rollout_loop(config, vllm_engine, vllm_inputs: List[Dict], prompts, multi_modal_inputs, sampling_params, vl_model_path, tool_version='video_toolbox', **kwargs):
    """
    Args:
        vllm_inputs (List[Dict]): The input prompt string containing placeholders like &lt;image&gt; or &lt;video&gt;.
    Example:
        vllm_inputs = [
            {
                #
                # >>> self.tokenizer.encode("&lt;image&gt;Please describe the scene.", add_special_tokens=False),
                "prompt_token_ids": [101, 12345, 67890, ...],
                "multi_modal_data": {
                    "image": [PIL.Image.open("example.png")]
                }
            },
        ]
    """
    from vllm.distributed import parallel_state as vllm_ps

    #! 能在sh里改
    agent_sampling_params = sampling_params.clone()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1 #! agent_sampling_params.n是vllm推理参数，与grpo里的n无关
    agent_sampling_params.include_stop_str_in_output = True
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    agent_sampling_params.max_tokens = max_generated_tokens

    # support custom stop specified in dataset, like </search>, ```, etc.
    custom_stop = list(config.agent.custom_stop)
    if custom_stop:
        prev_stop = sampling_params.stop if sampling_params.stop else []
        agent_sampling_params.stop = prev_stop + custom_stop
        print(f' [DEBUG stop] {type(prev_stop)=}, {type(custom_stop)=}, {type(agent_sampling_params.stop)=}')
    # Refer to: https://github.com/vllm-project/vllm/issues/1728
    # and https://github.com/vllm-project/vllm/issues/15976
    # def process_bad_tokens(token_ids, logits, exclude_token_ids=[]):
    #     for token_id in exclude_token_ids:
    #         logits[token_id] = -9999.999
    #     return logits

    # # NOTE: tmp for visual agent!
    # exclude_func = partial(process_bad_tokens, exclude_token_ids=[
    #     151643,    # <|endoftext|>
    #     151644,    # <|im_start|>
    # ])
    # agent_sampling_params.logits_processors = [exclude_func]
    # agent_sampling_params.bad_words = ["<|endoftext|>", "<|im_start|>"]

    tokenizer = hf_tokenizer(vl_model_path) #! 看相关的代码行有没有需要改的地方
    processor = hf_processor(vl_model_path)

    # support custom logit_bias specified for rollout, like '<|image_pad|>', '<|video_pad|>', set in ./verl/trainer/config/ppo_trainer.yaml
    custom_bias = list(config.agent.custom_bias)
    if custom_bias:
        prev_bias = sampling_params.logit_bias or {}
        bias_token_ids = tokenizer.convert_tokens_to_ids(custom_bias)
        new_bias = {bias_id: -100 for bias_id in bias_token_ids}
        agent_sampling_params.logit_bias = {**prev_bias, **new_bias}
        # print(f'[DEBUG logit_bias] merged {len(prev_bias)} old + {len(new_bias)} new → total {len(agent_sampling_params.logit_bias)}')

    if multi_modal_inputs is not None:
        multi_modal_inputs = multi_modal_inputs.tolist()
    else:
        multi_modal_inputs = [{}] * len(vllm_inputs)

    batch_size = len(vllm_inputs)
    print(f"[DEBUG] agent_rollout_loop started with batch_size: {batch_size}")
    print(f"[DEBUG] vllm_inputs[0] keys: {vllm_inputs[0].keys() if vllm_inputs else 'Empty'}")
    # print(vllm_inputs[1])
    
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    running_attn_masks = []
    reward_tensor_list = []
    reward_tensor_masks = []
    active_mask = []
    mm_input_list = []
    tool_call_cnt_list = []

    env = ParallelEnv(config.agent, tokenizer, processor)
    env.reset(prompts, vllm_inputs, n=sampling_params.n, tool_version=tool_version) # 每个batch rollout n次, 每个trajectory上一次只生成一个action
    print(f"[DEBUG] Environment reset completed")

    # interleaving inputs if sampling_params.n > 1 #! grpo的多次采样,采样出多个response
    for i in range(batch_size):
        for _ in range(sampling_params.n): # 这个n
            vllm_input_list.append(deepcopy(vllm_inputs[i]))
            prompt_ids = prompts.batch['input_ids'][i, :].clone()
            running_states.append(prompt_ids) # bsz*n 下同
            prompt_mask = prompts.batch['attention_mask'][i, :].clone() #!是个下三角的causual mask
            running_action_masks.append(prompt_mask)
            running_attn_masks.append(prompt_mask)
            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor) #! env reward, List[tensor]
            reward_tensor_mask = torch.zeros_like(prompt_ids, dtype=torch.bool)
            reward_tensor_masks.append(reward_tensor_mask)
            active_mask.append(True) #! 判断是否继续inference
            mm_input_list.append(deepcopy(multi_modal_inputs[i]))
            tool_call_cnt_list.append(0)
    
    print(f"[DEBUG] After initialization: len(running_states)={len(running_states)}, len(active_mask)={len(active_mask)}, active_count={sum(active_mask)}")

    # pg = vllm_ps.get_tp_group() # pg 进程组
    max_total_length = config.prompt_length + config.response_length


    video_path_list = [vinput['multi_modal_data'].pop('video_path', None) for vinput, is_active in zip(vllm_input_list, active_mask)]
    question_list = [vinput['multi_modal_data'].pop('question', None) for vinput, is_active in zip(vllm_input_list, active_mask)]


    for step in range(config.agent.max_turns): 
        print(f'[DEBUG 000] turn={step+1}, total={batch_size}, n={sampling_params.n}, num_active={sum(active_mask)}')
        if sum(active_mask) == 0:
            break

        active_video_path_list = [video_path for video_path, is_active in zip(video_path_list, active_mask) if is_active]
        active_question_list = [question for question, is_active in zip(question_list, active_mask) if is_active]

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]

        active_vllm_inputs = [
            {k: v for k, v in vinput.items() if k not in ['video_path', 'duration', 'question']}
            for vinput, is_active in zip(vllm_input_list, active_mask)
            if is_active
        ]

        print(f"========> [DEBUG] [active_vllm_inputs] {len(active_vllm_inputs)} active inputs, {active_vllm_inputs[0].keys() if active_vllm_inputs else 'None'}")
        # print(f"--------> [DEBUG] [decoded prompt_token_ids] {kwargs['tokenizer'].decode(active_vllm_inputs[0]['prompt_token_ids']) if active_vllm_inputs else 'None'}")

        # # #! temp save active_vllm_inputs, DELETE after debug
        # tempath = './active_vllm_inputs.pth'
        # import os
        # if not os.path.exists(tempath):
        #     torch.save(active_vllm_inputs, tempath)

        actions = vllm_engine.generate(
            prompts=active_vllm_inputs, # vllm会为list里的prompt独立调用采样策略 size bsz*n
            sampling_params=agent_sampling_params, # mei每次采样agent_sampling_params.n==1个
            use_tqdm=False
        ) # List[vllm.RequestOutput]
        # RequestOutput: #!NOTE, can not __getitem__
        #     request_id = request_id
        #     prompt = prompt
        #     prompt_token_ids = prompt_token_ids
        #     multi_modal_placeholders = multi_modal_placeholders or {}
        #     prompt_logprobs = prompt_logprobs
        #     outputs = outputs  : list[CompletionOutput]
        #     finished = finished
        #     metrics = metrics
        #     lora_request = lora_request
        #     encoder_prompt = encoder_prompt
        #     encoder_prompt_token_ids = encoder_prompt_token_ids
        #     num_cached_tokens = num_cached_tokens
        # print(f"========> [DEBUG] [step] {step+1}, {len(actions)} actions generated, {actions[0].outputs[0].token_ids if actions else 'None'}")
        for i in range(len(actions)):
            print(f"--------> [DEBUG] [decoded action token_ids] {kwargs['tokenizer'].decode(actions[i].outputs[0].token_ids) if actions else 'None'}")


        # if pg.is_first_rank:  
        #     # 具体来说, step就是在执行工具
        #     obs_results = env.step(active_indices, actions, video_path_list=active_video_path_list, question_list=active_question_list) # 四项: observations, rewards, dones, info, 前三个都是list, 每个list长度为bsz*n
        # else:
        #     obs_results = None

        # obs_results = pg.broadcast_object(obs_results)
        # import os
        # import torch.distributed as dist
        # rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"])
        # print(f"[DEBUG] begin broadcast, rank={rank}, world_size={world_size}")
        # def broadcast_data(obs_results, rank):
        #     # 创建一个空的 list 用于广播
        #     obj_list = [obs_results if rank == 0 else None]  # 只有 rank 0 有数据，其他为 None
        #     dist.broadcast_object_list(obj_list, src=0)  # 从 rank 0 广播
        #     return obj_list[0]  # 返回广播后的数据
        
        # if rank == 0:
        #     obs_results = env.step(active_indices, actions, video_path_list=active_video_path_list, question_list=active_question_list) # 四项: observations, rewards, dones, info, 前三个都是list, 每个list长度为bsz*n
        # else:
        #     obs_results = None
        # obs_results = broadcast_data(obs_results, rank)  # 广播数据到所有进程

        obs_results = env.step(active_indices, actions, video_path_list=active_video_path_list, question_list=active_question_list) # 四项: observations, rewards, dones, info, 前三个都是list, 每个list长度为bsz*n
        observations, rewards, dones, info = obs_results # bsz*n   # List[Dict], , , 

        for idx, obs, act, rew, done in zip(active_indices, observations, actions, rewards, dones): # 对于bsz*n个结果
            # process response token/tools token/action token ids 
            # obs: Dict

            response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64, device=running_states[idx].device) #? act.outputs结构?
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            #! ⬆️: running_states是个递归的,每个循环里把新的拼上去,相当于一个全局上下文了
            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                vllm_input_list[idx]['prompt_token_ids'], 
                response_token_ids,
                tokenizer=tokenizer,
            ) 

            action_reward = torch.zeros_like(response_token_ids, dtype=torch.float, device=reward_tensor_list[idx].device)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward]) 
            reward_tensor_list[idx][-1] += rew #! 把执行tools带来的reward 拼上去 #! 将env_reward也变成稀疏的token_level的

            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_action_masks[idx].device) #! tools里的token是可以相互看到的
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], action_mask]) #?这俩不是一样的嘛?
            action_reward_mask = torch.zeros_like(response_token_ids, dtype=torch.bool, device=reward_tensor_list[idx].device)
            action_reward_mask[-1] = True
            reward_tensor_masks[idx] = torch.cat([reward_tensor_masks[idx], action_reward_mask]) 

            # Ensure the last token is not obs
            #! max_total_length是计的token数
            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                active_mask[idx] = False
                continue #! 不用break是为了把mask的值敷上去

            if done or step == config.agent.max_turns - 1:
                active_mask[idx] = False
                continue #! 不用break是为了把mask的值敷上去
            tool_call_cnt_list[idx] += 1 #! 把有action的地方变为1

            # process obs tokens and images
            if 'prompt_token_ids_vllm' in obs.keys() and 'prompt_token_ids_model' in obs.keys():
                obs_token_ids_vllm = obs['prompt_token_ids_vllm']
                # print(f'running_states[idx] (type): {running_states[idx]}\n running_states[idx]: {running_states[idx]}')
                # print(f' [DEBUG] 6666666:{obs_token_ids_vllm.shape}, {obs["prompt_token_ids_model"]}')
                obs_token_ids_model = obs['prompt_token_ids_model'].to(running_states[idx].device) #! ERROR: obs['prompt_token_ids_model'] now is a str

                if len(vllm_input_list[idx]['prompt_token_ids']) + len(obs_token_ids_vllm) >= max_total_length:
                    active_mask[idx] = False
                    continue
                if running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length: #! 提前判断能不能塞进模型
                    active_mask[idx] = False
                    continue

                vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input( #
                    vllm_input_list[idx]['prompt_token_ids'], 
                    obs_token_ids_vllm,
                    tokenizer=tokenizer,
                )

                running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])
                obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=reward_tensor_list[idx].device)
                reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

                obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=running_action_masks[idx].device)
                running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=running_attn_masks[idx].device)
                running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])
                reward_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.bool, device=reward_tensor_list[idx].device)
                reward_tensor_masks[idx] = torch.cat([reward_tensor_masks[idx], reward_mask], dim=-1)

                if 'multi_modal_data' not in vllm_input_list[idx].keys(): # * TOCHECK, ctf
                    vllm_input_list[idx]['multi_modal_data'] = {}
                mm_data = obs.get('multi_modal_data', {})
                if 'image' in mm_data.keys() and mm_data['image']:
                    if 'image' not in vllm_input_list[idx]['multi_modal_data'].keys():
                        vllm_input_list[idx]['multi_modal_data']['image'] = []
                    vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']
                if 'video' in mm_data.keys():
                    if 'video' not in vllm_input_list[idx]['multi_modal_data'].keys():
                        vllm_input_list[idx]['multi_modal_data']['video'] = [] # TODO maybe a default video when use this modal
                    vllm_input_list[idx]['multi_modal_data']['video'] += mm_data['video'] # * TOCHECK = or +=
                mm_input = obs.get('multi_modal_inputs', {})
                if mm_input:
                    mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input)

            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                active_mask[idx] = False

    # if pg.is_first_rank:
    #     tools_call_list = env.tools_call_list
    #     tools_call_list = _pad_tool_lists(tools_call_list, pad_token="<PAD>", max_len=config.agent.max_turns)
    #     print(f'[DEBUG] 🙌 tools_call_list: {tools_call_list}')
    # else:
    #     tools_call_list = None
    # tools_call_list = pg.broadcast_object(tools_call_list)
    tools_call_list = env.tools_call_list
    tools_call_list = _pad_tool_lists(tools_call_list, pad_token="<PAD>", max_len=config.agent.max_turns)
    print(f'[DEBUG] 🙌 tools_call_list: {tools_call_list}')

    env.close()

    target_device = prompts.batch['input_ids'].device
    running_states = [state[: max_total_length] for state in running_states]
    
    # 添加调试信息和保护措施
    print(f"[DEBUG] running_states length: {len(running_states)}")
    print(f"[DEBUG] batch_size: {batch_size}, sampling_params.n: {sampling_params.n}")
    
    if not running_states:
        print("[DEBUG] Error: running_states is empty!")
        # 返回一个空的张量而不是 None
        return torch.empty(0, config.response_length, dtype=torch.long, device=target_device)
    
    state_tensor = pad_2d_list_to_length(running_states, tokenizer.pad_token_id, max_total_length).to(target_device) #! List with pad_token_id -> torch.Tensor
    
    if state_tensor is None or state_tensor.numel() == 0:
        print("[DEBUG] Error: state_tensor is empty or None!")
        return torch.empty(0, config.response_length, dtype=torch.long, device=target_device)

    # running_action_masks = [mask[: max_total_length] for mask in running_action_masks]
    # action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)

    # running_attn_masks = [mask[: max_total_length] for mask in running_attn_masks]
    # attn_mask_tensor = pad_2d_list_to_length(running_attn_masks, 0, max_total_length).to(target_device)

    # #* CHECK, gz
    # if processor is not None and processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
    #     # For Qwen-VL: (n*bs, 3, seq_len)
    #     position_ids_list = [
    #         get_rope_index(
    #             processor,
    #             input_ids=state_tensor[i, :],
    #             image_grid_thw=mm_input_list[i].get("image_grid_thw", None),
    #             video_grid_thw=mm_input_list[i].get("video_grid_thw", None),
    #             second_per_grid_ts=mm_input_list[i].get("second_per_grid_ts", None),
    #             attention_mask=attn_mask_tensor[i, :],
    #         ) for i in range(batch_size * sampling_params.n)
    #     ]
    #     position_ids_tensor = torch.stack(position_ids_list, dim=0)
    # else:
    #     # For LM: (n*bs, seq_len)
    #     position_ids_tensor = compute_position_id_with_mask(attn_mask_tensor)

    # reward_tensor_list = [reward[: max_total_length] for reward in reward_tensor_list]
    # reward_tensor = pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)

    # reward_tensor_masks = [reward_mask[: max_total_length] for reward_mask in reward_tensor_masks]
    # reward_mask_tensor = pad_2d_list_to_length(reward_tensor_masks, 0, max_total_length).to(target_device)

    # tool_call_tensor = torch.tensor(tool_call_cnt_list, dtype=torch.float32).to(target_device).unsqueeze(1)
    # invalid_tokens = {"<PAD>", "no_tool_call"}
    # valid_counts = [
    #     sum(token not in invalid_tokens for token in row)
    #     for row in tools_call_list
    # ]
    # valid_tool_call_tensor = torch.tensor(valid_counts, dtype=torch.float32).to(target_device).unsqueeze(1)
    print(f"[DEBUG] [state_tensor length] : {state_tensor.shape[-1]}, [config.response_length] : {config.response_length}")
    return state_tensor[:, -config.response_length: ]
    # return state_tensor #* CHECK


# DataProto.from_dict(
#         tensors={
#             "response": state_tensor[:, -config.response_length: ],
#             "action_mask": action_mask_tensor,
#             "attention_mask": attn_mask_tensor,
#             "position_ids": position_ids_tensor,
#             "env_reward": reward_tensor[:, -config.response_length: ],
#             "env_reward_mask": reward_mask_tensor[:, -config.response_length: ],
#             "tool_cnt": tool_call_tensor,
#             "valid_tool_call": valid_tool_call_tensor,
#         },
#         non_tensors={"multi_modal_inputs": mm_input_list, "tool_index": tools_call_list} if processor is not None else None
#     )