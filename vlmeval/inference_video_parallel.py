import torch
import torch.distributed as dist
import os
import csv
import warnings
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from typing import List, Dict, Any
import concurrent.futures
from tqdm import tqdm
import numpy as np

FAIL_MSG = 'Failed to obtain answer via API.'


def batch_infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, 
                    api_nproc=4, use_vllm=False, batch_size=8):
    """
    使用批量推理优化的数据推理函数
    """
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # Build model if needed
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        # 对于 API 模型，仍然使用原有的并行调用方式
        assert world_size == 1
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model

    # 检查是否支持批量推理
    supports_batch = hasattr(model, 'generate_batch') or (use_vllm and hasattr(model, 'llm'))
    
    if supports_batch and batch_size > 1:
        # 使用批量推理
        return batch_infer_with_vllm(model, model_name, dataset, sample_indices_subrem, 
                                   sample_map, res, out_file, batch_size, verbose)
    else:
        # 回退到串行推理
        return serial_infer_data(model, model_name, dataset, sample_indices_subrem, 
                               sample_map, res, out_file, verbose)


def batch_infer_with_vllm(model, model_name, dataset, sample_indices_subrem, 
                         sample_map, res, out_file, batch_size, verbose):
    """
    使用 vLLM 进行批量推理
    """
    dataset_name = dataset.dataset_name
    
    # 过滤掉已经处理的样本
    remaining_indices = [idx for idx in sample_indices_subrem if idx not in res]
    
    if not remaining_indices:
        return model
    
    # 准备批量数据
    batches = []
    for i in range(0, len(remaining_indices), batch_size):
        batch_indices = remaining_indices[i:i + batch_size]
        batches.append(batch_indices)
    
    print(f"[DEBUG] Processing {len(remaining_indices)} samples in {len(batches)} batches with batch_size={batch_size}")
    
    for batch_num, batch_indices in enumerate(tqdm(batches, desc=f"Processing batches")):
        batch_structs = []
        batch_dataset_names = []
        valid_batch_indices = []
        
        # 准备批量输入
        for idx in batch_indices:
            if idx in res:  # 双重检查
                continue
                
            try:
                # 设置模型参数
                _set_model_parameters(model, model_name, dataset)
                
                # 获取当前样本的数据集名称
                current_dataset_name = dataset_name
                if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
                    current_dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
                
                # 构建 prompt
                if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(current_dataset_name):
                    if dataset.nframe == 0:
                        raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
                    struct = model.build_prompt(
                        dataset.data.iloc[sample_map[idx]], dataset=current_dataset_name, 
                        video_llm=getattr(model, 'VIDEO_LLM', False)
                    )
                else:
                    struct = dataset.build_prompt(
                        sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
                    )
                
                batch_structs.append(struct)
                batch_dataset_names.append(current_dataset_name)
                valid_batch_indices.append(idx)
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error preparing sample {idx}: {e}")
                print(f"Full traceback:\n{error_details}")
                res[idx] = f"Failed to prepare: {e}"
        
        if not batch_structs:
            continue
            
        print(f"[DEBUG] Batch {batch_num+1}/{len(batches)}: Processing {len(batch_structs)} samples")
        
        # 执行批量推理
        try:
            if hasattr(model, 'generate_batch'):
                print(f"[DEBUG] Using model.generate_batch for {len(batch_structs)} samples")
                responses = model.generate_batch(batch_structs, batch_dataset_names)
            else:
                print(f"[DEBUG] Fallback to individual generate calls")
                # 对于使用 vLLM 的模型，可以尝试批量调用
                responses = []
                for struct, ds_name in zip(batch_structs, batch_dataset_names):
                    response = model.generate(message=struct, dataset=ds_name)
                    responses.append(response)
            
            # 验证响应数量
            if len(responses) != len(valid_batch_indices):
                print(f"[WARNING] Response count mismatch: got {len(responses)}, expected {len(valid_batch_indices)}")
                # 如果数量不匹配，回退到逐个处理
                responses = []
                for struct, ds_name in zip(batch_structs, batch_dataset_names):
                    try:
                        response = model.generate(message=struct, dataset=ds_name)
                        responses.append(response)
                    except Exception as e:
                        responses.append(f"Failed: {e}")
            
            # 存储结果
            for idx, response in zip(valid_batch_indices, responses):
                res[idx] = response
                if verbose:
                    print(f"[BATCH] idx {idx}: {response}", flush=True)
            
            # 定期保存结果
            if (batch_num + 1) % 5 == 0 or batch_num == len(batches) - 1:
                dump(res, out_file)
                print(f"[DEBUG] Saved progress after batch {batch_num+1}")
                
        except Exception as e:
            print(f"Batch inference failed for batch {batch_num+1}: {e}")
            print(f"[DEBUG] Falling back to individual processing for this batch")
            # 回退到串行处理这个批次
            for idx, struct, ds_name in zip(valid_batch_indices, batch_structs, batch_dataset_names):
                if idx in res:
                    continue
                try:
                    response = model.generate(message=struct, dataset=ds_name)
                    res[idx] = response
                    if verbose:
                        print(f"[INDIVIDUAL] idx {idx}: {response}", flush=True)
                except Exception as e2:
                    print(f"Failed to process sample {idx}: {e2}")
                    res[idx] = f"Failed to process: {e2}"
        
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
    
    # 最终保存结果
    final_res = {k: res[k] for k in remaining_indices if k in res}
    dump(res, out_file)
    print(f"[DEBUG] Final save completed. Processed {len(final_res)}/{len(remaining_indices)} samples")
    return model


def serial_infer_data(model, model_name, dataset, sample_indices_subrem, 
                     sample_map, res, out_file, verbose):
    """
    串行推理数据（原有逻辑的优化版本）
    """
    dataset_name = dataset.dataset_name
    
    for i, idx in tqdm(enumerate(sample_indices_subrem), total=len(sample_indices_subrem)):
        if idx in res:
            continue
            
        # 设置模型参数
        _set_model_parameters(model, model_name, dataset)
        
        # 构建 prompt
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            if dataset.nframe == 0:
                raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
            struct = model.build_prompt(
                dataset.data.iloc[sample_map[idx]], dataset=dataset, 
                video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        else:
            struct = dataset.build_prompt(
                sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
            )

        # 生成响应
        try:
            response = model.generate(message=struct, dataset=dataset_name)
        except Exception as e:
            print(f"Failed to process sample {idx}: {e}")
            response = f"{FAIL_MSG}: {type(e)} {str(e)}"
        
        torch.cuda.empty_cache()

        if verbose:
            print(f"idx {idx}: {response}", flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in sample_indices_subrem}
    dump(res, out_file)
    return model


def _set_model_parameters(model, model_name, dataset):
    """
    设置模型参数（从原有代码中提取）
    """
    if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
        if dataset.nframe > 0:
            if getattr(model, 'nframe', 0) != dataset.nframe:
                print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                setattr(model, 'nframe', dataset.nframe)
        elif getattr(model, 'fps', 0) == 0:
            raise ValueError(f'fps is not suitable for {model_name}')
        else:
            setattr(model, 'nframe', None)
            
    if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
        if dataset.fps > 0:
            if getattr(model, 'fps', 0) != dataset.fps:
                print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                setattr(model, 'fps', dataset.fps)
        elif getattr(model, 'nframe', 0) == 0:
            raise ValueError(f'nframe is not suitable for {model_name}')
        else:
            setattr(model, 'fps', None)
            
    if (
        'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
            print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
        if getattr(model, 'fps', None) is None and dataset.fps > 0:
            print(f'using {model_name} default setting for video, dataset.fps is ommitted')


def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    """
    API 模型的推理函数（从原有代码中复制）
    """
    dataset_name = dataset.dataset_name
    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    
    indices = list(samples_dict.keys())
    structs = []
    for i in indices:
        sample_idx = samples_dict[i]
        if getattr(dataset, 'pack', False):
            struct = dataset.videos[sample_idx]
        else:
            struct = dataset.build_prompt(sample_idx, video_llm=True)
        structs.append(struct)

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data_job_video_parallel(
        model,
        work_dir,
        model_name,
        dataset,
        result_file_name,
        verbose=False,
        api_nproc=4,
        use_vllm=False,
        batch_size=8):
    """
    并行优化的视频数据推理作业函数
    """
    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, result_file_name)
    
    # 如果结果文件已存在，直接返回
    if osp.exists(result_file):
        return model

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{osp.splitext(result_file_name)[0]}.pkl')
    out_file = tmpl.format(rank)

    # 使用批量推理
    model = batch_infer_data(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc,
        use_vllm=use_vllm,
        batch_size=batch_size)

    if world_size > 1:
        dist.barrier()

    # 合并结果
    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [data_all[x] for x in data['index']]

        if osp.splitext(result_file_name)[1] == '.tsv':
            dump(data, result_file, quoting=csv.QUOTE_NONE, sep='\t', escapechar='\\')
        else:
            dump(data, result_file)

    return model
