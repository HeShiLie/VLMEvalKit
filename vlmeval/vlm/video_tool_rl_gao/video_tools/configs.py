from omegaconf import OmegaConf

CONFIGS = OmegaConf.create({
    'load_format': "safetensors",
    # 'dtype',
    'enforce_eager': True,
    'gpu_memory_utilization': 0.98,
    'disable_log_stats': True,
    'enable_chunked_prefill': True,
    'seed': 0
    })
MCONFIGS = OmegaConf.create({
    'max_num_batched_tokens': 128000,
    'response_length': 118000,
    'max_prompt_length': 8192, #! same with prompt_length
    'prompt_length': 8192,
    'truncation': 'error',
    'agent':{
        'max_vllm_images': 40,
        'max_vllm_videos': 8,
        'activate_agent': True,
        'single_response_max_tokens': 10240,
        'concurrent_workers': 2,
        'show_tqdm': True,
        'max_turns': 6, # todo change back to 6
        'custom_stop': [],
        'custom_bias': ['<|image_pad|>', '<|video_pad|>']
    }
})

VIDEO_TOOL_SYSTEM_PROMPT_1 = "You are a helpful assistant.\n\n# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"video_image_retriever_tool\",\"description\":\"If the video is too long, use this tool to cut the videos into small clips and select topk clips that most correlated to the user's given question.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"topk\":{\"type\":\"int\",\"items\":{\"type\":\"number\"},\"minItems\":1,\"maxItems\":5,\"description\":\"The number of most correlated clips to be selected. If there are multiple suspicious clips, the value should be as large as possible, not exceeding \"maxItems\", and at least \"maxItems\" suspicious clips should be selected.\"}},\"required\":[\"topk\"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n**Example**: \n<tool_call> \n{\"name\": \"video_image_retriever_tool\", \"arguments\": {\"topk\": 2}} \n</tool_call>"

VIDEO_TOOL_SYSTEM_PROMPT_2 = "\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"video_perceiver_tool\",\"description\":\"If there's several clips but still hard to answer the question, use this tool to first select the most question-correlated clip and the most correlated frame. To do so, you need to point out which clip is the most correlated one and which frame is the most correlated one with idx in the form of numbers.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"clip_idx\":{\"type\":\"int\",\"items\":{\"type\":\"number\"},\"minItems\":1,\"maxItems\":1,\"description\":\"The idx of the most correlated clip compared with the question, not exceeding \"maxItems\", and at least \"maxItems\" suspicious clips should be selected.\"},\"frame_idx\":{\"type\":\"int\",\"description\":\"The idx of the most correlated frame in the selected clip.\"}},\"required\":[\"clip_idx\",\"frame_idx\"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n**Example**: \n<tool_call> \n{\"name\": \"video_perceiver_tool\", \"arguments\": {\"clip_idx\": 2, \"frame_idx\": 5}} \n</tool_call>"


VIDEO_TOOL_SYSTEM_PROMPT_3 = "\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"video_frame_grounder_tool\",\"description\":\"After focusing on specific frame by using other tools, zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"bbox_2d\":{\"type\":\"array\",\"items\":{\"type\":\"number\"},\"minItems\":4,\"maxItems\":4,\"description\":\"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.\"},\"label\":{\"type\":\"string\",\"description\":\"The name or label of the object in the specified bounding box (optional).\"}},\"required\":[\"bbox\"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n**Example**: \n<tool_call> \n{\"name\": \"video_frame_grounder_tool\", \"arguments\": {\"bbox_2d\": [10, 20, 100, 200], \"label\": \"the apple on the desk\"}} \n</tool_call>"

VIDEO_TOOL_SYSTEM_PROMPT = VIDEO_TOOL_SYSTEM_PROMPT_1 + VIDEO_TOOL_SYSTEM_PROMPT_2 + VIDEO_TOOL_SYSTEM_PROMPT_3

MCQ_SYS = """Carefully watch the video and pay attention to the cause and sequence of events, \
the detail and movement of objects, and the action and pose of persons. \
Based on your observations, select the best option that accurately addresses the question.
"""
SYS_PROMPT = MCQ_SYS + VIDEO_TOOL_SYSTEM_PROMPT