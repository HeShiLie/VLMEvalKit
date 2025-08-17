from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat
from ...dataset import DATASET_TYPE
from .video_tools.configs import CONFIGS, MCONFIGS, SYS_PROMPT
from .video_tools.inference_loop import agent_rollout_loop
from vllm import LLM, SamplingParams
from ...dataset import DATASET_MODALITY
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
from typing import List
import torch
import logging
from .qwen_utils import compute_position_id_with_mask
from verl import DataProto
import verl.utils.torch_functional as verl_F

from torchdata.stateful_dataloader import StatefulDataLoader

import numpy as np
from collections import defaultdict

from time import time

VLLM_MAX_IMAGE_INPUT_NUM = 24

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

# copied from deepeyes
def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

def _build_messages_into_dataproto(processor, tokenizer, message, text, images, videos, video_path, question, return_raw_chat=False, return_raw_dict=False, **kwargs):
    row_dict={}
    multi_modal_data = {} 
    origin_multi_modal_data = {}

    # print(f'=======> [DEBUG] [video_tool_rl.py] begin to build messages into dataproto, \n=======>  [video_path] {video_path}, \n=======>  [question] {question} \n =======> [text] {text} \n =======> [images] {images}, \n =======> [videos[0] shape] {videos[0].shape}')

    model_inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")

    if images is not None:
        origin_multi_modal_data['image'] = images
        multi_modal_data['image'] = images

    if videos is not None:
        multi_modal_data["video"] = [video.numpy() for video in videos]

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
    row_dict['origin_multi_modal_data'] = origin_multi_modal_data #! 之所以叫ori是因为只有一个input了
    row_dict["multi_modal_data"] = multi_modal_data 
    row_dict["multi_modal_inputs"] = dict(model_inputs) #! 调tool会传进去,就是processor处理后的格式,而且都直接变成id的形式了


    row_dict["multi_modal_data"]['video_path'] = video_path
    # row_dict["multi_modal_data"]['duration'] = get_video_duration_cv2(row_dict['extra_info']['video_path'])
    # 顺便把'question'键也补上吧
    row_dict["multi_modal_data"]['question'] = question

    # second_per_grid_ts isn't used for training, just for mrope
    row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=kwargs.get('configs', {}).get('max_prompt_length', 11240),
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=kwargs.get('configs', {}).get('truncation', 'error'),
    )

    if processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
        from verl.models.transformers.qwen2_vl import get_rope_index

        position_ids = [
            get_rope_index(
                processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]  # (1, 3, seq_len)
    else:
        position_ids = compute_position_id_with_mask(attention_mask)

    row_dict["input_ids"] = input_ids[0]
    row_dict["attention_mask"] = attention_mask[0]
    row_dict["position_ids"] = position_ids[0]

    max_prompt_length = kwargs.get('configs', {}).get('max_prompt_length', 11240)
    truncation=kwargs.get('configs', {}).get('truncation', 'right')

    raw_prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(raw_prompt_ids) > max_prompt_length:
        if truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-max_prompt_length :]
        elif truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: max_prompt_length]
        elif truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")

    row_dict["raw_prompt_ids"] = raw_prompt_ids
    # encode prompts without chat template
    if return_raw_chat:
        row_dict["raw_prompt"] = message

    # add index for each prompt
    index = row_dict.get("extra_info", {}).get("index", 0)
    row_dict["index"] = index

    # print(f"[DEBUG] [row_dict.keys] {row_dict.keys()}")
    # 找到键和值类型的最大长度
    # max_key_length = max(len(str(k)) for k in row_dict.keys())
    # max_type_length = max(len(str(type(v))) for v in row_dict.values())

    # 格式化打印，等号对齐
    # for k, v in row_dict.items():
        # print(f"[DEBUG] [row_dict] [key] {k:<{max_key_length}}       :      [type(value)] {str(type(v)):<{max_type_length}}")

    if not return_raw_dict:
        return DataProto.from_single_dict(collate_fn([row_dict]))
    else:
        return DataProto.from_single_dict(collate_fn([row_dict])), row_dict

def _get_info_from_list(messages: List[dict]):
    """
    Extracts video path and question from a list of messages.
    """
    for message in messages:
        if isinstance(message, dict):
            if message.get('type', 'text') == 'video':
                video_path = message.get('value', None)
            if message.get('type', 'text') == 'text':
                question = message.get('value', None)
        elif isinstance(message, str):
            # If the message is a string, we assume it contains the question.
            question = message
            video_path = None

    print(f"[DEBUG] [video_tool_rl.py] video_path: {video_path}, question: {question}")
    return video_path, question

# todo DONE: in configs.py, realize CONFIGS and MCONFIG
class VideoRLQwen(Qwen2VLChat):
    def __init__(self, model_path, vllm_config= CONFIGS, model_config = MCONFIGS, min_pixels = None, max_pixels = None, total_pixels = None, max_new_tokens=2048, top_p=0.001, top_k=1, temperature=0.01, repetition_penalty=1, use_custom_prompt = True, system_prompt = None, post_process = False, verbose = False, use_audio_in_video = False, use_vllm = True, tool_version='video_toolbox', **kwargs):
        super().__init__(model_path, min_pixels, max_pixels, total_pixels, max_new_tokens, top_p, top_k, temperature, repetition_penalty, use_custom_prompt, system_prompt, post_process, verbose, use_audio_in_video, use_vllm_but_no_init=True, **kwargs)
        self.model_path = model_path
        self.tool_version = tool_version
        # ! must use vlm
        # todo: add transformers
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 8:
            tp_size = 8
        elif gpu_count >= 4:
            tp_size = 4
        elif gpu_count >= 2:
            tp_size = 2
        else:
            tp_size = 1
        logging.info(
            f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
        )

        max_num_batched_tokens = model_config.get("max_num_batched_tokens", 8192)
        load_format = "dummy" if vllm_config.load_format.startswith("dummy") else vllm_config.load_format

        print(f"[DEBUG] [video_tool_rl.py] begin to init vllm engine, [gpu_memory_utilization] {vllm_config.gpu_memory_utilization}, [tp_size] {tp_size}")
        print(f"[DEBUG] FVCK [self.model_path] {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tp_size,
            # distributed_executor_backend="external_launcher",
            # dtype=vllm_config.dtype,
            enforce_eager=vllm_config.enforce_eager,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            # limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            # max_model_len=max_model_len + 16384,
            max_model_len=32768, #/4096
            load_format=load_format,
            disable_log_stats=vllm_config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=vllm_config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=False,
            seed=vllm_config.get("seed", 0),
            
            **({
                "limit_mm_per_prompt": dict(
                    image=model_config.agent.max_vllm_images, 
                    video=model_config.agent.max_vllm_videos,
                ),
            } if model_config.agent.activate_agent and model_config.agent.max_vllm_images else {})
        )
        print(f"[DEBUG] [video_tool_rl.py] SUCCESSFULLY init vllm engine")
        self.system_prompt = system_prompt if system_prompt is not None else SYS_PROMPT
        print(f"[DEBUG] [video_tool_rl.py] [self.system_prompt] {self.system_prompt}")
        self.sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.max_new_tokens, stop_token_ids=None
        ) #todo DONE
        self.model_config = model_config

    # def use_custom_prompt(self, dataset: str) -> bool:
    #     from vlmeval.dataset import DATASET_TYPE
    #     dataset_type = DATASET_TYPE(dataset, default=None)

    #     if not self._use_custom_prompt:
    #         return False
    #     if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
    #         return True
    #     if dataset_type == 'MCQ':
    #         if dataset is not None and 'LEGO' in dataset:
    #             return False
    #         return True
    #     if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
    #         return True
    #     if dataset_type == 'VQA' and dataset not in {'MMVet'}:  # MMVet VQA has it's own prompt
    #         return True
    #     if dataset in {'MVBench'}:
    #         return True
    #     return False
    
    # def build_prompt(self, line, dataset, **kwargs) -> list[dict[str, str]]:
    #     from vlmeval.dataset import DATASET_TYPE

    #     # Handle both dataset object and dataset name string
    #     if hasattr(dataset, 'dataset_name'):
    #         dataset_name = dataset.dataset_name
    #     elif isinstance(dataset, str):
    #         dataset_name = dataset
    #     else:
    #         dataset_name = str(dataset)

    #     if dataset_name in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
    #         return self._build_mmmu_prompt(line, dataset_name)
    #     # elif dataset_name in {'MVBench', 'MVBench_MP4_8frame'}:
    #     #     return self._build_mvbench_prompt(line, dataset_name)
    #     dataset_type = DATASET_TYPE(dataset_name, default=None)
    #     if dataset_type == 'MCQ':
    #         return self._build_mcq_prompt(line, dataset_name)
    #     if dataset_type == 'Y/N':
    #         return self._build_yorn_prompt(line, dataset_name)
    #     if dataset_type == 'VQA':
    #         return self._build_vqa_prompt(line, dataset_name)
    #     raise ValueError(f'Unsupported dataset: {dataset_name}')
    
    
    def generate_batch(self, messages_list, dataset_list=None):
        """
        批量生成函数，充分利用vLLM的批量推理能力
        """
        from vllm import SamplingParams
        from .video_tools.inference_loop import agent_rollout_loop
        from vlmeval.utils.memory_utils import MemoryMonitor, force_memory_cleanup

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        with MemoryMonitor("Batch Generation", verbose=True):
            batch_inputs = []
            batch_results = []

            prompts_list = []

            print(f"[DEBUG] {len(messages_list)=} messages to process in batch")
        
        for i, (message, dataset) in enumerate(zip(messages_list, dataset_list or [None] * len(messages_list))):
            try:
                messages = []
                if self.system_prompt is not None:
                    messages.append({'role': 'system', 'content': self.system_prompt})
                messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
                
                print(f"[DEBUG] {len(messages)=} for message {i}")

                if self.verbose:
                    print(f'\033[31m{messages}\033[0m')

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = process_vision_info(messages)

                if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
                    assert len(videos) == 1
                    # videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]

                    video_path, question = _get_info_from_list(message)
                    prompts, prompts_row = _build_messages_into_dataproto(
                        self.processor, self.processor.tokenizer, message, text, 
                        images, videos, video_path, question, configs=MCONFIGS, return_raw_dict=True
                    ) 
                    prompts_list.append(prompts_row)
                    
                    # 准备vLLM输入
                    idx = prompts.batch["input_ids"]
                    batch_size = idx.size(0)

                    non_tensor_batch = prompts.non_tensor_batch
                    if "raw_prompt_ids" not in non_tensor_batch:
                        non_tensor_batch["raw_prompt_ids"] = np.array(
                            [_pre_process_inputs(self.processor.tokenizer.pad_token_id, idx[j]) for j in range(batch_size)], 
                            dtype=object
                        )

                    if "multi_modal_data" in non_tensor_batch:
                        for raw_prompt_ids, multi_modal_data in zip(
                            non_tensor_batch["raw_prompt_ids"], non_tensor_batch["multi_modal_data"]
                        ):
                            vllm_input = {"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data}
                            if isinstance(vllm_input["prompt_token_ids"], np.ndarray):
                                vllm_input["prompt_token_ids"] = vllm_input["prompt_token_ids"].tolist()
                            batch_inputs.append(vllm_input)
                    else:
                        for raw_prompt_ids in non_tensor_batch["raw_prompt_ids"]:
                            vllm_input = {"prompt_token_ids": raw_prompt_ids}
                            if isinstance(vllm_input["prompt_token_ids"], np.ndarray):
                                vllm_input["prompt_token_ids"] = vllm_input["prompt_token_ids"].tolist()
                            batch_inputs.append(vllm_input)
                
                elif images:
                    vllm_input = {
                        "prompt": text,
                        "multi_modal_data": {"image": images},
                    }
                    batch_inputs.append(vllm_input)
                else:
                    vllm_input = {"prompt": text}
                    batch_inputs.append(vllm_input)
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error preparing input {i}: {e}")
                print(f"Full traceback:\n{error_details}")
                batch_inputs.append(None)
        
        print("----------->", len(prompts_list), "inputs prepared for batch processing")
        
        # 保留原始的prompts_list用于回退处理
        original_prompts_list = prompts_list.copy()
        
        # Create StatefulDataLoader and get the batched data
        prompts_list_batch = StatefulDataLoader(prompts_list, collate_fn=collate_fn, batch_size=len(prompts_list), drop_last=False)
        batch_prompts = next(iter(prompts_list_batch))
        
        # batch_prompts is now a DataProto-compatible batched dictionary
        # We need to convert it to DataProto format for agent_rollout_loop
        from verl import DataProto
        prompts_dataproto = DataProto.from_single_dict(batch_prompts)
        
        print("==========>", f"Batch created with shape: {batch_prompts['input_ids'].shape if 'input_ids' in batch_prompts else 'unknown'}")


        # 过滤掉失败的输入
        valid_inputs = [(i, inp) for i, inp in enumerate(batch_inputs) if inp is not None]
        valid_indices = [i for i, inp in valid_inputs]
        valid_vllm_inputs = [inp for i, inp in valid_inputs]
        
        if not valid_vllm_inputs:
            return ["Failed to prepare any valid inputs"] * len(messages_list)
        
        try:
            # 执行批量推理
            if any('multi_modal_data' in inp and 'video' in str(inp.get('multi_modal_data', {})) 
                   for inp in valid_vllm_inputs):
                # 对于包含视频的输入，使用agent_rollout_loop的批量推理能力
                try:
                    # 收集所有视频相关的输入进行真正的批量推理
                    video_inputs = [inp for inp in valid_vllm_inputs 
                                  if 'multi_modal_data' in inp and 'video' in str(inp.get('multi_modal_data', {}))]
                    non_video_inputs = [inp for inp in valid_vllm_inputs 
                                      if not ('multi_modal_data' in inp and 'video' in str(inp.get('multi_modal_data', {})))]
                    
                    outputs = []
                    
                    # 批量处理视频输入
                    if video_inputs:
                        # print(video_inputs)
                        print(f"[DEBUG] Processing {len(video_inputs)} video inputs in batch")
                        batch_outputs = agent_rollout_loop(
                            config=self.model_config,
                            vllm_engine=self.llm,
                            vllm_inputs=video_inputs,  # 传入整个列表进行批量推理
                            prompts=prompts_dataproto,  # 使用正确的DataProto格式
                            multi_modal_inputs=None,
                            sampling_params=self.sampling_params,
                            vl_model_path=self.model_path, 
                            tokenizer=self.processor.tokenizer,
                            tool_version=self.tool_version,
                        )
                        
                        print(f"[DEBUG] batch_outputs shape: {batch_outputs.shape if hasattr(batch_outputs, 'shape') else type(batch_outputs)}")
                        
                        # agent_rollout_loop 返回的是 (batch_size, response_length) 的张量
                        # 需要解码每个样本的输出
                        if batch_outputs is None:
                            print("[DEBUG] Error: batch_outputs is None!")
                            outputs = ["Failed to process: batch_outputs is None"] * len(video_inputs)
                        elif isinstance(batch_outputs, torch.Tensor):
                            if batch_outputs.numel() == 0:
                                print("[DEBUG] Error: batch_outputs is empty tensor!")
                                outputs = ["Failed to process: empty tensor"] * len(video_inputs)
                            else:
                                for i in range(batch_outputs.shape[0]):
                                    decoded_output = self.processor.tokenizer.decode(
                                        batch_outputs[i], skip_special_tokens=True, clean_up_tokenization_spaces=False
                                    )
                                    outputs.append(decoded_output)
                        else:
                            # 如果不是张量，可能是列表
                            try:
                                for output in batch_outputs:
                                    decoded_output = self.processor.tokenizer.decode(
                                        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                    )
                                    outputs.append(decoded_output)
                            except Exception as e:
                                print(f"[DEBUG] Error decoding batch_outputs: {e}")
                                outputs = [f"Failed to decode: {e}"] * len(video_inputs)
                    
                    # 处理非视频输入（如果有的话）
                    if non_video_inputs:
                        print(f"[DEBUG] Processing {len(non_video_inputs)} non-video inputs")
                        non_video_outputs = self.llm.generate(non_video_inputs, self.sampling_params)
                        outputs.extend([o.outputs[0].text for o in non_video_outputs])
                        
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Batch video inference failed: {e}")
                    print(f"Full traceback:\n{error_details}")
                    # 回退到逐个处理
                    outputs = []
                    for i, inp in enumerate(valid_vllm_inputs):
                        try:
                            if 'multi_modal_data' in inp and 'video' in str(inp.get('multi_modal_data', {})):
                                # 为单个样本创建DataProto
                                # 从原始prompts_list中获取第i个样本
                                single_prompt_dict = original_prompts_list[i] if i < len(original_prompts_list) else original_prompts_list[0]
                                single_prompt_dataproto = DataProto.from_single_dict(collate_fn([single_prompt_dict]))
                                
                                output = agent_rollout_loop(
                                    config=self.model_config,
                                    vllm_engine=self.llm,
                                    vllm_inputs=[inp],  # 单个输入
                                    prompts=single_prompt_dataproto,  # 传入对应的单个prompt的DataProto格式
                                    multi_modal_inputs=None,
                                    sampling_params=self.sampling_params,
                                    vl_model_path=self.model_path, 
                                    tokenizer=self.processor.tokenizer,
                                    tool_version=self.tool_version,
                                )
                                # output 是一个形状为 (1, response_length) 的张量
                                if isinstance(output, torch.Tensor):
                                    decoded_output = self.processor.tokenizer.decode(
                                        output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                                    )
                                else:
                                    decoded_output = self.processor.tokenizer.decode(
                                        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                    )
                                outputs.append(decoded_output)
                            else:
                                output = self.llm.generate([inp], self.sampling_params)
                                outputs.append(output[0].outputs[0].text)
                        except Exception as e2:
                            import traceback
                            error_details = traceback.format_exc()
                            print(f"Individual inference failed: {e2}")
                            print(f"Full traceback:\n{error_details}")
                            outputs.append(f"Failed to process: {e2}")
            else:
                # 对于图像或纯文本输入，可以使用vLLM的批量推理
                outputs = self.llm.generate(valid_vllm_inputs, self.sampling_params)
                outputs = [o.outputs[0].text for o in outputs]
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Batch inference failed: {e}")
            print(f"Full traceback:\n{error_details}")
            # 回退到逐个处理
            outputs = []
            for inp in valid_vllm_inputs:
                try:
                    output = self.llm.generate([inp], self.sampling_params)
                    outputs.append(output[0].outputs[0].text)
                except Exception as e2:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Individual inference failed: {e2}")
                    print(f"Full traceback:\n{error_details}")
                    outputs.append(f"Failed to process: {e2}")
        
        # 重新组织结果
        results = ["Failed to process"] * len(messages_list)
        for idx, output in zip(valid_indices, outputs):
            results[idx] = output
        
        # 清理内存
        del batch_inputs, valid_vllm_inputs, outputs
        if 'prompts_dataproto' in locals():
            del prompts_dataproto
        if 'batch_prompts' in locals():
            del batch_prompts
        if 'original_prompts_list' in locals():
            del original_prompts_list
        force_memory_cleanup(verbose=True)
            
        if self.verbose:
            for i, result in enumerate(results):
                print(f'\033[32mBatch {i}: {result}\033[0m')
                
        return results
    
    def generate_inner(self, message, dataset=None):
        from vllm import SamplingParams

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        start_time = time()

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        print(f'[DEBUG] [time] finishing process vision info in vllm. time cost: {time() - start_time:.2f}s')

        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]

            video_inputs = {
                "prompt": text[0],
                "multi_modal_data": {"video": videos_nd[0]},
                "mm_processor_kwargs":{}
            }
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print(f'video input sequence may be too long for vllm (longer than {VLLM_MAX_IMAGE_INPUT_NUM}), Maybe cannot generate response for VLLM')
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.max_new_tokens, stop_token_ids=None
        )
        if images:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": images},
                },
                sampling_params=sampling_params,
            )
        elif videos_nd:

            video_path, question = _get_info_from_list(message)
            prompts = _build_messages_into_dataproto(self.processor, self.processor.tokenizer, message, text, images, videos, video_path, question, configs=MCONFIGS) 
            idx = prompts.batch["input_ids"]
            batch_size = idx.size(0)

            non_tensor_batch = prompts.non_tensor_batch
            if "raw_prompt_ids" not in non_tensor_batch: #? raw_prompt_ids 对应的是个str还是?
                non_tensor_batch["raw_prompt_ids"] = np.array(
                    [_pre_process_inputs(self.processor.tokenizer.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
                )

            if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
                raise RuntimeError("vllm sharding manager is not work properly.")

            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                for raw_prompt_ids, multi_modal_data in zip(
                    non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
                ):
                    vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
            else:
                vllm_inputs = [
                    {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
                ]

            # ensure the type of `prompt_token_ids` passed to vllm is list[int]
            # https://github.com/volcengine/verl/pull/772
            for input_data in vllm_inputs:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
                elif not isinstance(input_data["prompt_token_ids"], list):
                    raise TypeError(
                        f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                    )
            outputs = agent_rollout_loop(
                config=self.model_config,
                vllm_engine=self.llm,
                vllm_inputs=vllm_inputs, 
                prompts=prompts,
                multi_modal_inputs=non_tensor_batch.get("multi_modal_inputs", None),
                sampling_params=self.sampling_params,
                vl_model_path=self.model_path, 
                tokenizer = self.processor.tokenizer,
                tool_version=self.tool_version,
            )
            
            # outputs 是一个形状为 (batch_size, response_length) 的张量
            if isinstance(outputs, torch.Tensor):
                # 取第一个样本的输出（因为这里是单个推理）
                output_tokens = outputs[0]
            else:
                # 如果不是张量，可能是列表，取第一个元素
                output_tokens = outputs[0] if isinstance(outputs, list) and len(outputs) > 0 else outputs
                
            out = self.processor.tokenizer.decode(
                output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out
            
            # 清理视频相关的内存
            del videos, videos_nd, vllm_inputs, prompts, output_tokens
            if 'non_tensor_batch' in locals():
                del non_tensor_batch
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            print(f'[DEBUG] [video_tool_rl.py] response: {response}')
            return response
        else:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                },
                sampling_params=sampling_params,
            )

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        
        # 清理内存
        if 'images' in locals():
            del images
        if 'videos' in locals():
            del videos
        if 'text' in locals():
            del text
        if 'messages' in locals():
            del messages
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return generated_text