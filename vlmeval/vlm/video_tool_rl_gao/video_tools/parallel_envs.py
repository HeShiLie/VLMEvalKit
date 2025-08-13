import torch
from tqdm import tqdm
import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video
from typing import Union, Optional
from io import BytesIO
import io
from pathlib import Path
import os

from .tool_envs import ToolBase

def find_file(search_dir: Path, filename: str) -> Optional[Path]:
    """递归查找文件
    
    Args:
        search_dir: 搜索目录
        filename: 目标文件名
        
    Returns:
        找到返回Path对象，未找到返回None
    """
    try:
        for root, _, files in os.walk(search_dir):
            if filename in files:
                return Path(root) / filename
        return None
    except OSError as e:
        print(f"搜索文件出错: {e}")
        return None
    
def process_image(image: Union[dict, Image.Image]) -> Image.Image:
    # if isinstance(image, dict) and 'bytes' in image.keys():
    #     image_object = Image.open(BytesIO(image['bytes']))

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    return fetch_image(image)

def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    **kwargs: Optional[dict]
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

    # print(f"[DEBUG] [process_video] : {video}")

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError('video must be a dict with "video" key')
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    #todo: nextqa--done change the key of video path into the real path
    data_source = kwargs.get("data_source", None)
    data_path = kwargs.get("data_path", None)
    # dataset_name = video.get("data_source",None)
    if data_source == 'NextQA':
        video_name = str(video.get("video", None))+'.mp4'
        # 从data_path 去grab出绝对位置
        video_path = find_file(data_path+'/nextqa', video_name)
        video["video"] = f"file://{video_path}" 

    elif not data_source:
        video["video"] = f"file://{video['video']}" 

    print(f"[DEBUG] video['video'] [in RLHFDataset process_video]: {video['video']}")

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    return fetch_video(video)

def _preprocess_multi_modal_inputs(prompt_str, processor, **kwargs): #todo: done 需要改,看看怎么处理(process)视频clips
    """
    Preprocesses multi-modal inputs (images/videos) into token ids for use with vLLM and Qwen2-style models.

    This function prepares the prompt by replacing special multimodal tokens,
    processes image input into bytes, and constructs a model-ready input dictionary.

    Args:
        prompt_str (str): The input prompt string containing placeholders like <image> or <video>.
        processor (transformers.Processor): A processor (e.g., AutoProcessor) used to convert multimodal inputs.
        **kwargs: Additional keyword arguments, expecting:
            - multi_modal_data (dict): A dictionary with optional "image" (list of PIL.Image) keys.
            - tool_name (str, optional): Tool name used for debugging/logging.

    Returns:
        vllm_input_prompt (str): The transformed prompt string with special tokens replaced for vLLM.
        input_ids (torch.Tensor): The tokenized input IDs ready for model input.
        mm_inputs (dict): Additional multimodal tensors (e.g., pixel_values) returned by the processor.

    Example:
        >>> vllm_input_prompt, input_ids, mm_inputs = _preprocess_multi_modal_inputs(
        ...     prompt_str="<image> Please describe the scene.",
        ...     processor=AutoProcessor.from_pretrained(...),
        ...     multi_modal_data={"image": [PIL.Image.open("example.png")]}
        ... )
    """
    if processor is None or "multi_modal_data" not in kwargs:
        return prompt_str, prompt_str, {}

    # preprocess obs prompt to chat template like
    prompt_str = prompt_str.replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>')
    vllm_input_prompt = prompt_str.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')

    # preprocess obs mm data
    input_mm_data = kwargs.get("multi_modal_data", {"image": []})
    input_mm_info = kwargs.get("vision_infos", {"image": []})

    ## preprocess obs image data    
    image_info_list = []
    for img in tqdm(input_mm_data.get("image", []), desc="[DEBUG] [_preprocess_multi_modal_inputs] Prerocessing images"):
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        buf.close()
        img_info = {"bytes": png_bytes}
        image_info_list.append(img_info)
    input_mm_data["image"] = [process_image(img) for img in image_info_list]
    
    ## preprocess obs video data 
    video_info_list = []
    for video_path in tqdm(input_mm_info.get("video", []), desc="[DEBUG] [_preprocess_multi_modal_inputs] Prerocessing videos"):
        if video_path.startswith("file://"):
            raw_path = video_path[len("file://"):]
        else:
            raw_path = video_path
        import os
        abs_path = os.path.abspath(raw_path)

        video_info = {
            "type": "video",
            "video": abs_path,
            "fps": 2,
            "min_frames": 2,
            "max_frames": 8
        }
        video_info_list.append(video_info)
    input_mm_data["video"] = [process_video(video) for video in video_info_list]
    # if input_mm_data["video"]:
    #     input_mm_data["video"] = input_mm_data["video"][-1]

    ## return preprocessed obs mm data to kwargs #! due to shallow copy
    kwargs["multi_modal_data"] = {
        "image": input_mm_data["image"],
        "video": input_mm_data["video"]
    }

    print('[DEBUG] vllm_input_prompt in _preprocess_multi_modal_inputs: (length)', len(vllm_input_prompt))
    try:
        model_inputs = processor(
            text=[vllm_input_prompt], 
            images=input_mm_data["image"] or None, 
            videos=input_mm_data["video"] or None,
            return_tensors="pt")
    except Exception as e:
        print(f"[DEBUG] Exception in [FUNC:_preprocess_multi_modal_inputs]\n")
        print(f"[DEBUG] [tool name]: {kwargs.get('tool_name', 'unknown')}\n")
        print(f"[DEBUG] [input_mm_data]: {input_mm_data}\n")
        print(f"[DEBUG] [input_mm_data['video']: length]: {len(input_mm_data['video']) if isinstance(input_mm_data['video'], list) else 'N/A'}\n")
        print(f"[DEBUG] [vllm_input_prompt]: {vllm_input_prompt}\n")
        print(f"[ERROR] {type(e).__name__}: {e}\n")
        raise 

    input_ids = model_inputs.pop("input_ids")[0]
    print(f'[DEBUG] || [FUNC:_preprocess_multi_modal_inputs] || {input_ids.shape=}, {input_ids.dtype=}, {input_ids.device=}')
    attention_mask = model_inputs.pop("attention_mask")[0]
    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")
    mm_inputs = dict(model_inputs)

    return vllm_input_prompt, input_ids, mm_inputs #! chat_template_str, torch.tensor, processor_dict.pop("input_ids","attention_mask")

def _strip_system_block(text: str) -> str:
    """
    删除 text 中第一个 <|im_start|>system ... <|im_end|> 区块（含标签），
    并返回删除后的字符串。
    如果找不到匹配的开始或结束标签，则返回原文。
    """
    # 非贪婪匹配，匹配跨行
    pattern = r"<\|im_start\|>system.*?<\|im_end\|>"
    # 替换为空
    result = re.sub(pattern, "", text, flags=re.S)
    return result

def execute_tool_call(sample, tokenizer=None, processor=None, pbar=None, **kwargs):
    action_string = sample.get('action', '')
    tool = sample.get('tool', None)

    # non-agent data
    if action_string == '' or tool is None:
        print(f"[Debug] noooooooooooooo toooooools, action_string: {action_string}, tool: {tool}")
        return {}, 0.0, True, {}

    print(f"[Debug] tool [{tool.name}] executed")
    tool_result, reward, done, info = tool.execute(action_string, **kwargs) 
    #! 假如action_string里格式不符，但是数据集里给了env_name, 这里就返回"", 0.0, True, {}
    print(f"[Debug] type tool_result [{type(tool_result)}]")
    # post-process
    if not tool_result:
        tool_result_info = {}

    elif isinstance(tool_result, str):#? 这不应该是个字典吗,这是调用了什么tool啊?
        # Format 1: text output
        obs_token_ids = tokenizer.encode(tool_result, add_special_tokens=False)
        tool_result_info = {
            "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
            "prompt_token_ids_model": torch.tensor(obs_token_ids),
        }

    elif isinstance(tool_result, list) and isinstance(tool_result[0], dict):
        # Format 2: [{"role": "...", "content": "..."}, ...]
        obs_token_ids = tokenizer.apply_chat_template(tool_result, add_generation_prompt=True, return_tensors='pt')[0]

        # NOTE: skip the sp (and the \n token that comes after it) added by Qwen tokenizer
        eos_start_idx = torch.nonzero(obs_token_ids == tokenizer.eos_token_id)
        if eos_start_idx.shape[0] > 0:
            eos_start_idx = eos_start_idx[0].item()
            obs_token_ids = obs_token_ids[eos_start_idx + 1 : ]
        else:
            raise ValueError(f"tool [{tool.name}] returned type List[str] output must be in openai/qwen format : {tool_result}")

        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids,
            "prompt_token_ids_model": obs_token_ids,
        }

    elif isinstance(tool_result, dict):
        #!"prompt"和"chat"是一回事,有的版本用"chat" 有的版本用“prompt”
        # Format 3: {"prompt": "...", "chat": [{"role": "...", "content": "..."}, ...], "multi_modal_data": ...}
        prompt_str = tool_result.pop("prompt", "") #! tool_result is {"prompt": ,"multi_modal_data":{"image":, "video_clip_paths":,}}
        chat_list = tool_result.pop("chat", [])

        if len(prompt_str) == 0 and len(chat_list) == 0:
            raise ValueError("Both prompt_str and chat_list are invalid")
        elif len(prompt_str) == 0 and len(chat_list) > 0:
            prompt_str = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True, tokenize=False)
            prompt_str = _strip_system_block(prompt_str)

        prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(prompt_str, processor, **tool_result) 
        obs_token_ids_vllm = tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors='pt')[0]
        # print(f' [DEBUG] 2222222222: {type(obs_token_ids_model)}')

        if isinstance(obs_token_ids_model, str):
            obs_token_ids_model = tokenizer.encode(obs_token_ids_model, add_special_tokens=False, return_tensors='pt')[0]

        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids_vllm, # Token_ids; original <vision_start><pad><vision_end>
            "prompt_token_ids_model": obs_token_ids_model, # Token_ids; padding <vision_start> vision tokens... <vision_end>
            **tool_result   #! "multi_modal_data"
        }
        if mm_inputs:
            tool_result_info["multi_modal_inputs"] = mm_inputs

    else:
        raise ValueError(f"Invalid tool_result type: {type(tool_result)=} -- {tool_result}")

    if pbar is not None:
        pbar.update(1)
    return tool_result_info, reward, done, info

class ParallelEnv:
    """
    The interface is designed to be the similar to : https://github.com/openai/gym
    """
    def __init__(self, env_config, tokenizer, processor, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer
        self.processor = processor

        # type: List[ Dict[ Str, ToolBase subclasses ] ]
        self.tools = []
        self.tools_call_state_list = [] # 预期是list of list, 共bsz*n个子list, 每个子list是该条trajectory上的工具调用情况

    def _get_state_from(self, info):
        '''
        Return
            form refer to "info" in execute_tool_call
            for example:
                if there a tool call (no matter success or not), return a Dict as follows:
                    {
                        "status": "success", 
                        "tool_used": tool_name, 
                        "info_stated": info_need_to_stated
                    }
                else, return:
                    {
                        "status": "fail",
                        "tool_used": "no_tool_call", 
                        "info_stated": None
                    }
        '''
        print(f' [DEBUG] _get_state_from info.keys(): {info.keys()}')
        if info.get('status', None) == 'success':
            return info
        else:
            return {
                "status": "fail",
                "tool_used": "no_tool_call", #! TGIGPO
                "info_stated": None
            }

    def step(self, active_indices, actions, **kwargs): # actions:应该是整个batch的
        """
        Input:
        - actions: vllm.RequestOutput #!!!!此处有误, actions 并非 vllm.RequestOutput, 而是 一个DataProto

        Output:
        - observations: List[Dict], content like {"prompt_token_ids": ..., "multi_modal_data": ...}, 
                multi_modal_data only appears when there are images/videos in obs
        - rewards: List[ float ].
                each time after an action being executed, procedure rewards can be assigned to 
                the last valid token of model outputs. This might be useful for ..., 
                e.g., invalid action, code execution error, format error,
                or video game envs where immediate feedback is available.
        - dones: List[ Boolean ]
        - infos: Dict, for debugging only
        """
        obs_list = [{}] * len(actions) # len 为 n*batch
        reward_list = [0.0] * len(actions)
        done_list = []
        valid_indices = []
        real_indices = []
        valid_actions = []

        # tools_call_state_list_single_turn = []
        valid_video_path_list = []
        valid_question_list = []

        active_video_path_list = kwargs.get('video_path_list', [None])
        active_question_list = kwargs.get('question_list', [None])

        # 1. filtering valid actions
        for i, (idx, act) in enumerate(zip(active_indices, actions)):
            if act.outputs[0].finish_reason == 'length':
                done_list.append(True)
                self.tools_call_state_list[idx].append('no_tool_call')
                continue

            if len(act.outputs[0].token_ids) == 0: #! 如果没有生成token, 直接跳过
                done_list.append(True)
                self.tools_call_state_list[idx].append('no_tool_call')
                continue

            done_list.append(False)
            real_indices.append(i) #! 储存真正的索引
            valid_indices.append(idx) #! 储存索引对应的值? #！储存继续inference的bsz*n列表里的索引
            valid_actions.append(act.outputs[0].text)
            valid_video_path_list.append(active_video_path_list[i])
            valid_question_list.append(active_question_list[i])
            # tools_call_state_list_single_turn.append("no_tool_call")

        agent_inputs = []
        for i, idx, action in zip(real_indices, valid_indices, valid_actions): # 是这一轮(turn)的 #!n*bsz中有效的action的个数
            agent_inputs.append(dict(
                idx=i,
                valid_idx=idx,
                action=action,
                tool=self.tools[idx], #! 在self.reset后赋予
            )) # 是这一轮(turn)的n*bsz中有效的action的个数

        # 2. executing actions (sync or async)
        num_workers = min(self.config.concurrent_workers, len(valid_actions))
        pbar = tqdm(total=len(valid_actions), desc=f'[Turn {len(self.tools_call_state_list[0])+1}] Tool calling on {num_workers} workers') if self.config.show_tqdm else None
        if num_workers <= 1: #?这还能==0?
            for i, agi in enumerate(agent_inputs): # agi是agent_inputs的一个元素,包含idx, valid_idx, action, tool #! 按照trajectory, 逐个执行
                subidx = agi['idx'] # 这个是在active list里的idx, 不一定连续

                current_kwargs = {}

                current_sample_video_path = valid_video_path_list[i]
                current_sample_question = valid_question_list[i]

                # 把self.tools_call_state_list 传给 execute_tool_call, 以实现成功的连续工具调用
                current_kwargs['video_path'] = current_sample_video_path
                current_kwargs['question'] = current_sample_question
                current_kwargs['tools_call_state_list'] = self.tools_call_state_list[agi['valid_idx']]
                print('hhhhhhh', current_kwargs['tools_call_state_list'], 'RANK: ',os.environ.get('RANK', 'no'))

                obs, reward, done, info = execute_tool_call(agi, self.tokenizer, self.processor, pbar=pbar, **current_kwargs) #! this info is only use to check

                # * CHECK 看看把工具调用状态记录在info里有无问题
                self.tools_call_state_list[agi['valid_idx']].append(self._get_state_from(info))
                self.tools_call_list[agi['valid_idx']].append(self._get_state_from(info)['tool_used'])
                print(f"[Debug] || [ParallelEnv.step] || Turn {len(self.tools_call_state_list[agi['valid_idx']])}: tools call state list of idx {agi['valid_idx']} in batch: {self.tools_call_state_list[agi['valid_idx']]}")
                # coding by ctf
                # tools_call_state_list_single_turn[subidx] = self._get_state_from(info)
                # # coding by gaozhe: record the current tool_state

                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done #! 如果execute_tool_call说已经结束了,就更新为结束了
        else:
            partial_tool_func = partial(execute_tool_call, tokenizer=self.tokenizer, processor=self.processor, pbar=pbar, **kwargs)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                raw_outputs = list(executor.map(partial_tool_func, agent_inputs))
            for agi, raw in zip(agent_inputs, raw_outputs):
                obs, reward, done = raw[0], raw[1], raw[2]
                subidx = agi['idx']
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done

        return obs_list, reward_list, done_list, {} # 每个list长度为bsz*n

    def reset(self, prompts, vllm_inputs, n=1, **kwargs):
        self.tools = []
        self.tools_call_state_list = []
        self.tools_call_list = [] # List[List[str]]

        reset_output_list = []
        assert len(prompts) == len(vllm_inputs), f"{len(prompts)=}, {len(vllm_inputs)=}"

        # 获取 tool_version 参数，默认为 'video_toolbox'
        tool_version = kwargs.get('tool_version', 'video_toolbox')

        num_agent, num_non_agent = 0, 0
        for i in range(len(prompts)):
            data_item = prompts[i]  # DataProtoItem
            tool_name = tool_version  # 使用传入的 tool_version 参数
            raw_prompt = data_item.non_tensor_batch.pop('raw_prompt', None) #! 小作文

            vllm_input_item = vllm_inputs[i]   # {"prompt_token_ids": ..., "multi_modal_data": ...}
            multi_modal_data = vllm_input_item.get("multi_modal_data", None)
            origin_multi_modal_data = data_item.non_tensor_batch.pop("origin_multi_modal_data", None)
            for _ in range(n):
                if tool_name:
                    # init tools from config field `tool_name_key`
                    #! 这里(ref to ToolBase)初始化的注册好的tools,它是个全局注册表
                    tool_fns = ToolBase.create(tool_name)
                    reset_output = tool_fns.reset(
                        raw_prompt=raw_prompt, 
                        multi_modal_data=deepcopy(multi_modal_data),
                        origin_multi_modal_data=deepcopy(origin_multi_modal_data),
                    )
                    self.tools.append(tool_fns) #! self.tools: List[Option[ToolBase, None]], len == bsz*rollout_n 
                    reset_output_list.append(reset_output)
                    num_agent += 1
                else:
                    # non-agent data
                    self.tools.append(None)
                    reset_output_list.append(None)
                    num_non_agent += 1
                self.tools_call_state_list.append([])
                self.tools_call_list.append([]) 
        
        print(f' [DEBUG agent] {num_agent=}, {num_non_agent=}')
        return reset_output_list

    def close(self):
        self.tools = []