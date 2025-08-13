import numpy as np
import copy
from .tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
import traceback

from .aux_models.languagebind import RETRIEVER_TOOL_PROMPT, Retrieval_Manager, preprocess_video_with_progress

from PIL import Image
import cv2

def get_video_duration_cv2(video_path):
    """
    使用 OpenCV 获取视频时长, 返回视频时长(秒),向下取整
    
    Args:
        video_path (str): 视频文件路径
        
    Returns:
        int: 视频时长（秒），如果失败返回 -1
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return -1
        
        # 获取帧率和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        cap.release()
        
        if fps > 0:
            duration = int(frame_count / fps)
            return duration
        else:
            return -1
            
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return -1

def check_previous_tool_call(tool_name, tools_call_state_list): # tools_call_state_list是个
    # idx 实际上是在这个bsz*n 里的index
    if tool_name == 'video_perceiver_tool':
        # 从后往前检查, 前面有没有调用过 video_image_retriever_tool
        for i in range(len(tools_call_state_list)-1, -1, -1):
            if tools_call_state_list[i]['info_stated']['tool_name'] == 'video_image_retriever_tool':
                return True

    else:
        return False

def get_ouput_from_previous_tool_call(tool_name, tools_call_state_list):
    # idx 实际上是在这个bsz*n 里的index
    if tool_name == 'video_perceiver_tool':
        # 对于这个 tools_call_state_list (共bsz*n个traj), 先把idx对应的那一串拿出来
        # 从后往前检查, 前面有没有调用过 video_image_retriever_tool
        for i in range(len(tools_call_state_list)-1, -1, -1):
            if tools_call_state_list[i]['info_stated']['tool_name'] == 'video_image_retriever_tool':
                return tools_call_state_list[i]['info_stated']['video_clip_paths']
    else:
        raise ValueError(f"Unsupported tool name: {tool_name}. Cannot retrieve output from previous tool call.")

class VideoToolBox(ToolBase):
    '''
    我们这个video tool box 包含了 retriever、perciever和Frame Grounder
    其中:
    Retriever: #
        Input:
            长视频
        Output:
            Video Clips
    perciever: #需修改
        Input:
            Video Clip
        Output:
            关键帧
    Frame Grounder:
        Input:
            关键帧
        Output:
            关键帧的grounding结果
    均为自己实现
    '''
    name = "video_toolbox" #CHANGE BACK AFTER DEBUG
    user_prompt = "Here is the observation returned after you executed the tool call." #todo: maybe we should change the prompt to let the model know how to use the tools

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current video being processed

    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.search(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer
    
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        if not tool_call_match:
            raise ValueError("No tool call found in the action string.")
        
        tool_call_json = tool_call_match.group(1).strip()
        try:
            tool_call = json.loads(tool_call_json)
            return tool_call
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool call: {e}")
        
    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        # print(f"[DEBUG] kwargs['video_path_list'][idx] {kwargs['video_path_list'][idx]}")
        current_image = None # * DEBUG
        current_video_paths = [] # * DEBUG

        try:
            answer = self.extract_answer(action_string)
            if answer:
                return "", 0.0, True, {}
            tool_call = self.extract_action(action_string)
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            
            #! here adding the customized tools
            if tool_name == "video_image_retriever_tool":
                # 1. turn the videos into a number of video clips
                # 2. select number of topk clips most relevant to the user qurey
                #! ⬆️ using OpenAI api
                # 3. get a list of video clip paths
                import omegaconf
                #todo: initialize args in the form of omegaconfig
                retriever_args = omegaconf.OmegaConf.create({
                    "retriever_type": "large",  # Assuming 'large' is the type of retriever you want to use
                    "clip_duration": 10,  # Duration of each clip in seconds
                    "dataset_folder": "",  # Path to the video dataset
                })
                
                clip_save_folder = "./temp/clip_save_folder"
                video_path = kwargs['video_path']  # Get the video path from kwargs
                question = kwargs['question']
                # duration = kwargs['duration_list']
                duration = get_video_duration_cv2(video_path)  # Get the video duration

                preprocess_video_with_progress(video_path, clip_save_folder)

                retriever = Retrieval_Manager(args=retriever_args, clip_save_folder=clip_save_folder+'/clips/10') #! 可能有隐患,有个llm在里面, 可能导致oom
                message = RETRIEVER_TOOL_PROMPT.format(clip_duration=1, MAX_DS_ROUND=10, question=question,duration=duration) #todo: copy from vdr

                # retriever.preprocess_video_with_progress(video_path)
                video_clip_paths = retriever.get_informative_clips(
                        message, video_path=video_path, top_k=args['topk'], total_duration=duration
                    ) # 返回list of strings
                del retriever

                current_image = [] # *  # No image to return for this tool
                current_video_paths = [video_clip_path[0] for video_clip_path in video_clip_paths]  # List of video clip paths
                # current_video_paths = [] # * TOCHECK
                
                info_need_to_stated = {
                    "tool_name": tool_name,
                    "video_clip_paths": video_clip_paths,
                    "video_path": video_path,
                    "duration": duration,
                }
  
            elif tool_name == "video_perceiver_tool": #! maybe problemetic
                # 从一个list里的video clip paths中选择 某个 clip,并指出哪一帧最接近
                clip_idx = args.get("clip_idx", 0)  # Default to the first clip if not specified
                if check_previous_tool_call(tool_name, kwargs['tools_call_state_list']): #kwargs['tools_call_state_list'] 是个list, 要么是‘no_tool_call’ 要么见info的格式
                    video_clip_paths = get_ouput_from_previous_tool_call(tool_name, kwargs['tools_call_state_list'])
                else:
                    raise ValueError("[DEBUG] tool ||{tool_name}|| No video clip paths available. Please run the video_image_retriever_tool first.")
                
                if args["clip_idx"] >= len(video_clip_paths):
                    raise ValueError(f"[DEBUG] tool ||{tool_name}|| clip_idx {args['clip_idx']} out of range for video_clip_paths with length {len(video_clip_paths)}.")
                selected_clip_path = video_clip_paths[clip_idx]
                # read the video clip
                import cv2
                cap = cv2.VideoCapture(selected_clip_path)
                if not cap.isOpened():
                    raise ValueError(f"[DEBUG] tool ||{tool_name}|| Cannot open video clip: {selected_clip_path}")
                frame_idx = args.get("frame_idx", 0)  # Default to the first frame if not specified
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()  # Read the specified frame
                if not success:
                    raise ValueError(f"[DEBUG] tool ||{tool_name}|| Failed to read frame {frame_idx} from video clip {selected_clip_path}.")
                # Convert the frame to a numpy array
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                frame = Image.fromarray(frame)  # Convert to PIL Image

                current_image = [frame]  # Wrap in a list to match expected format
                video_clip_paths = []  # Return the selected clip path

                info_need_to_stated = {
                    "tool_name": tool_name,
                    }

            elif tool_name == "video_frame_grounder_tool": 
                #
                bbox = args["bbox_2d"]
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][-1]
                cropped_img = img.crop(bbox)

                current_image = [cropped_img]

                info_need_to_stated = {
                    "tool_name": tool_name,
                    "bbox_2d": bbox,
                }

            elif tool_name == "video_browser_tool": #! 实际上这个工具调用优先级应当最高
                # 1. random select several frames from the entire video
                video_path = kwargs['video_path']
                num_frames = 15
                
                import cv2
                import random
                import numpy as np

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"[DEBUG] tool ||{tool_name}|| Cannot open video: {video_path}")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # 如果视频帧数少于需要的帧数，抛出异常
                if total_frames < num_frames:
                    raise ValueError(f"[DEBUG] tool ||{tool_name}|| Video has fewer frames ({total_frames}) than requested ({num_frames}).") #todo: maybe not raise ValueError
                
                # 随机选择一个起始点，确保后续 num_frames 帧不会超出总帧数
                start_frame = random.randint(0, total_frames - num_frames)

                # 创建 (T, C, H, W) 的张量
                video_tensor = np.zeros((num_frames, 3, 224, 224), dtype=np.uint8)

                for t in range(num_frames):
                    # 设置视频读取的位置
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + t)
                    success, frame = cap.read()  # 读取帧
                    if not success:
                        raise ValueError(f"[DEBUG] tool ||{tool_name}|| Failed to read frame {start_frame + t} from video.")

                    # 调整帧大小为指定的 frame_size (H, W)，并将通道顺序从 HWC 转为 CHW
                    resized_frame = cv2.resize(frame, (224, 224))
                    # print(f"[DEBUG] tool ||{tool_name}|| Resized frame shape: {resized_frame.shape}")
                    video_tensor[t] = np.transpose(resized_frame, (2, 0, 1))  # HWC -> CHW

                cap.release()  # 释放资源

                current_image = [Image.fromarray(video_tensor[i], mode = 'RGB') for i in range(video_tensor.shape[0])]  # Convert to list of images
                video_clip_paths = []

                info_need_to_stated = {
                    "tool_name": tool_name,
                    "video_path": video_path,
                    "num_frames": num_frames,
                }
    
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            
            # Prepare the observation
            #! 到时候tokenizer.encode, 他自己写的格式转换的代码,中间的说不定还不用改
            # * CHECK
            image_str = "<image>" if current_image else ""
            video_str = "<video>"*len(current_video_paths) if current_video_paths else ""
            # video_str = ""
            # TODO: support multi images/videos 
            obs = {
                "prompt": "<|im_end|>\n<|im_start|>user\n" + video_str + image_str + self.user_prompt + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [current_image[0]] if current_image else [], "video": None}, #  current_image: List[Image]
                # "vision_infos": {"image": None, "video": [current_video_paths[0]] if current_video_paths else []},
                "vision_infos": {"image": None, "video": current_video_paths if current_video_paths else []},
                "tool_name": tool_name, # TODO: delete after debug
            }
            reward = 0.5  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name, "info_stated": info_need_to_stated}
            print(f'[DEBUG] SUCCESS ACTION AT STEP {len(kwargs["tools_call_state_list"])} The str is {action_string=}')
            return obs, reward, done, info
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)}')
            print(f'[DEBUG] FAILED ACTION AT STEP {len(kwargs["tools_call_state_list"])} {action_string=}')
            print(f'[DEBUG] Traceback:\n{error_traceback}')
            obs = {
                "prompt": f"<|im_start|>user\nError: {str(e)}<|im_end|>\n<|im_start|>assistant\n",
                "multi_model_data": None,
            }
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info
        
    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data

if __name__ == "__main__":
    # Example usage (for testing)
    tool = VideoToolBox("video_toolbox", "Tool for video processing", {})
    tool.multi_modal_data={
        "video_path": "/mnt/gz/Data/ActivityNetQA/ActivityNetQA/all_test/v__-JNaelSKO8.mp4",  # Example video path and duration in seconds
        "duration": 28,  # Example video path and duration in seconds
        "image": None,
        "question": "What is the main activity in this video?",
    }
    
    video_image_retriever_action = """
    <tool_call>
    {"name": "video_image_retriever_tool", "arguments": {"topk": 1}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(video_image_retriever_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    print(obs)
    
    # video_perceiver_action = """
    # <tool_call>
    # {"name": "video_perceiver_tool", "arguments": {"clip_idx": 0, "frame_idx": 0}}
    # </tool_call>
    # """
    # obs, reward, done, info = tool.execute(video_perceiver_action)
    # print(f"Rotate result - Reward: {reward}, Info: {info}")
    
    # video_frame_grounder_action = """
    # <tool_call>
    # {"name": "video_frame_grounder_tool", "arguments": {"bbox": [0,0, 100, 100]}}
    # </tool_call>
    # """
    # obs, reward, done, info = tool.execute(video_frame_grounder_action)
    # print(f"Rotate result - Reward: {reward}, Info: {info}")

    # video_browser_action = """
    # <tool_call>
    # {"name": "video_browser_tool", "arguments": None}
    # </tool_call>
    # """
    # obs, reward, done, info = tool.execute(video_browser_action)
    # print(f"Rotate result - Reward: {reward}, Info: {info}")

    # # Test invalid JSON (should return reward=0.0)
    # invalid_action = """
    # <tool_call>
    # {"name": "video_image_retriever_tool", "arguments": {"image_path": "test.jpg", "angle": 90}
    # </tool_call>
    # """
    # obs, reward, done, info = tool.execute(invalid_action)
    # print(f"Invalid JSON result - Reward: {reward}, Info: {info}")
    
    # # Test unknown tool (should return reward=0.0)
    # unknown_tool_action = """
    # <tool_call>
    # {"name": "unknown_tool", "arguments": {"param": "value"}}
    # </tool_call>
    # """
    # obs, reward, done, info = tool.execute(unknown_tool_action)
    # print(f"Unknown tool result - Reward: {reward}, Info: {info}")