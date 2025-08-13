# 设置工作路径
import os
os.chdir("/workspace/codes/DeepEyes")
# 加载模型
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path="../../model_zoo/qw2.5vl-7b-instruct/"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# 设置工作路径
os.chdir("/workspace/codes/DeepEyes")
# 加载数据
from datasets import load_dataset
dataset = load_dataset("parquet", data_files="/workspace/Data/videoRL_moveto48/nextqa/train5k.parquet", split="train")

idx = 2001

# Preparation for inference
messages = dataset[idx]['prompt']
# print(dataset[0]['extra_info'].keys())
# print(dataset[0].keys())
# print(messages)
# print(len(messages))

# 使用绝对路径并添加角色字段
messages[1]['content'] = [
    {
        "type": "text",
        "text": '<video>' + messages[1]['content'].replace('<video>',''),
    },
    {
        "type": "video",
        'video': "file:///workspace/Data/videoRL_moveto48/nextqa/videos/1155/3696878746.mp4",
        'fps': 1, 
        'max_frames': 40, 
        'max_pixels': 12544, 
        'min_frames': 1, 
        'min_pixels': 3136, 
        'total_pixels': 262144
    }
]

NEW_SYS_POMPT = '''Think first, call **video_image_retriever_tool**, **video_perceiver_tool**, **video_frame_grounder_tool**, **video_browser_tool** if needed, then answer. Remember, the first 3 tools should be used in a sequential manner, while the last tool is used separately.  Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>'''
messages[0]['content'] += NEW_SYS_POMPT
# print(len(messages))

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
print(f"[DEBUG] [after process_vision_info] [inputs] [SYS Prompt]: {messages[0]['content']}")
print(f"[DEBUG] [after process_vision_info] [inputs] [USER Prompt]: {messages[1]['content'][0]['text']}")
inputs = inputs.to("cuda")

# check the answer
print(f"[Question] {dataset[idx]['extra_info']['question']}")
print(f"[Answer] {dataset[idx]['reward_model']['ground_truth']}")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=6800)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"[output_text] {output_text}")

# deepeyes
# 加载一条数据
deepeyes_data = load_dataset("parquet", data_files='../../Data/deepeyes/data_0.1.2_visual_toolbox_v2.parquet', split="train")

from PIL import Image
import io

idx_deepeyes = 107

deepeyes_data[idx_deepeyes]
messages_deepeyes = deepeyes_data[idx_deepeyes]['prompt']
# print(deepeyes_data[idx_deepeyes]['images'])
print(len(messages_deepeyes))
messages_deepeyes[1]['content'] = [
    {
        "type": "text",
        "text": messages_deepeyes[1]['content'] + ' a1: left, a2: right. choose one from the 2 answers',
    },
    {
        "type": "image",
        'image': Image.open(io.BytesIO(deepeyes_data[idx_deepeyes]['images'][0]['bytes'])),
    }
]

text = processor.apply_chat_template(
    messages_deepeyes, tokenize=False, add_generation_prompt=True
)
print(f"[DEBUG] [before process_vision_info] [message] [deepeyes]: {messages_deepeyes[1]['content'][0]['text']}")
image_inputs, video_inputs = process_vision_info(messages_deepeyes)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
print(f"[DEBUG] [after process_vision_info] [inputs] [deepeyes]: {messages_deepeyes[1]['content'][0]['text']}")

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1280)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"[output_text] {output_text}")

# 
print(f'[Question] {deepeyes_data[idx_deepeyes]["extra_info"]["question"]}')
print(f'[Answer] {deepeyes_data[idx_deepeyes]["extra_info"]["answer"]}')

# import matplotlib.pyplot as plt
import io
from PIL import Image

# 从二进制数据创建PIL图像
image = Image.open(io.BytesIO(deepeyes_data[idx_deepeyes]['images'][0]['bytes']))
print(f"[diverse content]: {messages[1]['content'][0]['text']}")
