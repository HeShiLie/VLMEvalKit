from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

if __name__ == "__main__":
    model_path = '/workspace/codes/DeepEyes/verl_checkpoints/agent_vlagent_0717/debug_for_single_node_qw2_5vl-7b-instruct/global_step_8/actor/hf_merged/'

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    gpu_count = torch.cuda.device_count()
    if gpu_count >= 8:
        tp_size = 8
    elif gpu_count >= 4:
        tp_size = 4
    elif gpu_count >= 2:
        tp_size = 2
    else:
        tp_size = 1

    active_vllm_inputs = torch.load('./active_vllm_inputs.pth', weights_only=False)

    llm = LLM(
        model=model_path,
        max_num_seqs=5,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1, "video": 1},
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=8192, stop_token_ids=None
    )

    # generate
    outputs = llm.generate(
                    prompts=active_vllm_inputs,
                    sampling_params=sampling_params,
                )

    print(f"========> [DEBUG]  {len(outputs)} actions generated, {outputs[0].outputs[0].token_ids if outputs else 'None'}")
    print(f"--------> [DEBUG] [decoded action token_ids] {tokenizer.decode(outputs[0].outputs[0].token_ids) if outputs else 'None'}")
