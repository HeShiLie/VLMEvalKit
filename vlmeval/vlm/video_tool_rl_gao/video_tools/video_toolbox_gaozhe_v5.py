import numpy as np
import copy
from .tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
import requests

import traceback
import omegaconf

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

    elif tool_name == 'video_frame_grounder_tool':
        # 从后往前检查, 前面有没有调用过 video_perceiver_tool
        for i in range(len(tools_call_state_list)-1, -1, -1):
            if tools_call_state_list[i]['info_stated']['tool_name'] == 'video_perceiver_tool':
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
    elif tool_name == 'video_frame_grounder_tool':
        # 对于这个 tools_call_state_list (共bsz*n个traj), 先把idx对应的那一串拿出来
        # 从后往前检查, 前面有没有调用过 
        for i in range(len(tools_call_state_list)-1, -1, -1):
            if tools_call_state_list[i]['info_stated']['tool_name'] == 'video_perceiver_tool':
                return tools_call_state_list[i]['info_stated']['selected_image_path']
    else:
        raise ValueError(f"Unsupported tool name: {tool_name}. Cannot retrieve output from previous tool call.")

def filter_short_clips(video_paths: list[str], min_frames: int = 30) -> list[str]:
    """
    过滤掉帧数小于等于 min_frames 的视频路径

    Args:
        video_paths (list[str]): 视频路径列表
        min_frames (int): 最小帧数要求（默认30）

    Returns:
        list[str]: 过滤后的路径列表
    """
    valid_paths = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"无法打开视频：{path}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count > min_frames:
            valid_paths.append(path)
        else:
            print(f"丢弃帧数 {frame_count} 的视频：{path}")

    return valid_paths


def outloading_retriver(video_path: str, clip_save_folder: str, question: str, topk: int, duration: int, 
                       service_url: str = "http://localhost:17860", max_retries: int = 5, 
                       retry_delay: float = 10.0, request_timeout: int = 300):
    """
    通过远程服务调用 Retrieval Manager 进行视频检索
    
    Args:
        video_path: 视频路径
        clip_save_folder: 片段保存目录
        question: 查询问题
        topk: 返回前k个结果
        duration: 视频时长
        service_url: 检索服务的URL
        max_retries: 最大重试次数（默认5次）
        retry_delay: 重试间隔时间（秒，默认10秒）
        request_timeout: 请求超时时间（秒，默认600秒）
        
    Returns:
        video_clip_paths: 视频片段路径列表
    """
    import requests
    import json
    import time
    
    # 重试机制主循环
    for attempt in range(max_retries + 1):
        try:
            # 方法1: 首先尝试使用 gradio_client（推荐方式）
            try:
                from gradio_client import Client
                
                print(f"[INFO] 使用 gradio_client 连接到检索服务: {service_url} (尝试 {attempt + 1}/{max_retries + 1})")
                client = Client(service_url)
                
                # 调用视频检索API
                result = client.predict(
                    video_path,
                    question,
                    topk,
                    duration,
                    clip_save_folder,
                    api_name="/predict_1"
                )
                
                if isinstance(result, dict) and result.get("success", False):
                    video_clip_paths = result["video_clip_paths"]
                    # 过滤短片段
                    video_clip_paths = filter_short_clips(video_clip_paths, min_frames=60)
                    print(f"[INFO] 检索服务返回 {len(video_clip_paths)} 个有效片段 (尝试 {attempt + 1} 成功)")
                    return video_clip_paths
                else:
                    error_msg = result.get("error", "gradio_client调用失败") if isinstance(result, dict) else "意外的返回格式"
                    
                    # 检查是否是服务忙碌错误
                    if "busy" in str(error_msg).lower() or "queue" in str(error_msg).lower() or "occupied" in str(error_msg).lower():
                        if attempt < max_retries:
                            print(f"[WARNING] 服务忙碌，{retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries + 1})")
                            time.sleep(retry_delay)
                            continue
                    
                    raise Exception(f"gradio_client调用失败: {error_msg}")
                    
            except ImportError:
                print("[WARNING] gradio_client 未安装，尝试HTTP请求方式")
                raise Exception("gradio_client 未安装")
            except Exception as e:
                # 检查是否是服务忙碌相关的错误
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["busy", "queue", "occupied", "timeout", "connection"]):
                    if attempt < max_retries:
                        print(f"[WARNING] [video_toolbox_gaozhe_v5.outloading_retriver] gradio_client 调用失败 (服务忙碌相关): {str(e)}")
                        print(f"[INFO] [video_toolbox_gaozhe_v5.outloading_retriver] {retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries + 1})")
                        time.sleep(retry_delay)
                        continue
                
                print(f"[WARNING] [video_toolbox_gaozhe_v5.outloading_retriver] gradio_client 调用失败 (非服务忙碌原因): {str(e)}")
                print(f"[INFO] [video_toolbox_gaozhe_v5.outloading_retriver] 尝试HTTP请求方式...")
                raise Exception(f"gradio_client 失败: {str(e)}")
        
        except Exception as gradio_error:
            # 方法2: 回退到HTTP请求方式
            try:
                print(f"[INFO] 尝试HTTP请求方式... (尝试 {attempt + 1}/{max_retries + 1})")
                
                # 尝试新的Gradio API端点
                success = False
                for api_endpoint in ["/gradio_api/predict_1", "/predict_1"]:
                    try:
                        api_url = f"{service_url}{api_endpoint}"
                        
                        response = requests.post(
                            api_url,
                            json={
                                "data": [video_path, question, topk, duration, clip_save_folder],
                                "session_hash": "retrieval_session"
                            },
                            headers={"Content-Type": "application/json"},
                            timeout=request_timeout
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # 处理Gradio返回格式
                            if "data" in result and len(result["data"]) > 0:
                                result_data = result["data"][0]
                                
                                # 如果返回的是字符串，尝试解析JSON
                                if isinstance(result_data, str):
                                    try:
                                        result_json = json.loads(result_data)
                                    except json.JSONDecodeError:
                                        result_json = {"success": False, "error": f"无法解析返回数据: {result_data}"}
                                elif isinstance(result_data, dict):
                                    result_json = result_data
                                else:
                                    result_json = {"success": False, "error": f"意外的返回数据类型: {type(result_data)}"}
                                    
                                if result_json.get("success", False):
                                    video_clip_paths = result_json["video_clip_paths"]
                                    video_clip_paths = filter_short_clips(video_clip_paths, min_frames=60)
                                    print(f"[INFO] HTTP请求成功，返回 {len(video_clip_paths)} 个有效片段 (尝试 {attempt + 1} 成功)")
                                    return video_clip_paths
                                else:
                                    error_msg = result_json.get("error", "HTTP请求返回错误")
                                    
                                    # 检查是否是服务忙碌错误
                                    if any(keyword in str(error_msg).lower() for keyword in ["busy", "queue", "occupied"]):
                                        print(f"[WARNING] 服务忙碌 ({api_endpoint}): {error_msg}")
                                        break  # 退出API端点循环，进行重试
                                    else:
                                        print(f"[WARNING] HTTP请求失败 ({api_endpoint}): {error_msg}")
                                        continue
                            else:
                                print(f"[WARNING] HTTP请求返回无效格式 ({api_endpoint})")
                                continue
                        elif response.status_code == 503:  # 服务不可用
                            print(f"[WARNING] 服务暂时不可用 ({api_endpoint}): {response.status_code}")
                            break  # 退出API端点循环，进行重试
                        else:
                            print(f"[WARNING] HTTP请求失败 ({api_endpoint}): {response.status_code} - {response.text}")
                            continue
                            
                    except requests.exceptions.Timeout:
                        print(f"[WARNING] HTTP请求超时 ({api_endpoint})")
                        break  # 退出API端点循环，进行重试
                    except requests.exceptions.ConnectionError:
                        print(f"[WARNING] 无法连接到服务 ({api_endpoint})")
                        break  # 退出API端点循环，进行重试
                    except requests.exceptions.RequestException as req_error:
                        print(f"[WARNING] HTTP请求异常 ({api_endpoint}): {str(req_error)}")
                        continue
                
                # 如果到这里说明所有HTTP端点都失败了，但可能是临时性的
                if attempt < max_retries:
                    print(f"[WARNING] 所有HTTP端点都无法访问，{retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries + 1})")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception("所有HTTP端点都无法访问")
                
            except Exception as http_error:
                error_str = str(http_error).lower()
                if any(keyword in error_str for keyword in ["busy", "queue", "occupied", "timeout", "connection"]) and attempt < max_retries:
                    print(f"[WARNING] HTTP请求失败 (可能服务忙碌): {str(http_error)}")
                    print(f"[INFO] {retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries + 1})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[ERROR] HTTP请求方式也失败了: {str(http_error)}")
                    if attempt >= max_retries:
                        raise Exception(f"检索服务不可用: gradio_client和HTTP请求都失败 - {str(http_error)}")
                    else:
                        time.sleep(retry_delay)
                        continue
    
    # 如果所有重试都失败了
    raise Exception(f"检索服务在 {max_retries + 1} 次尝试后仍然不可用，可能model pool全被占用")

class VideoToolBoxV5(ToolBase):
    name = "video_toolbox_v5" #CHANGE BACK AFTER DEBUG
    user_prompt = "Here is the observation returned after you executed the tool call." #todo: maybe we should change the prompt to let the model know how to use the tools

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current video being processed
        
        # 检索服务现在由 ParallelEnv 管理，这里不再需要启动
        print(f"📝 [VideoToolBoxV5] 检索服务由 ParallelEnv 管理")

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

                video_path = kwargs['video_path']  # Get the video path from kwargs
                question = args['retrieving_sentence']
                clip_save_folder = "./temp_v5/clip_save_folder"
                topk=args['topk']
                duration = get_video_duration_cv2(video_path)  # Get the video duration
                
                # 检查是否提供了服务URL参数
                service_url = kwargs.get('retrieval_service_url', 'http://localhost:17860')
                
                # 检索配置参数（可选）
                max_retries = kwargs.get('retrieval_max_retries', 5)
                retry_delay = kwargs.get('retrieval_retry_delay', 10.0)
                request_timeout = kwargs.get('retrieval_timeout', 600)

                video_clip_paths = outloading_retriver(
                    video_path, clip_save_folder, question, topk, duration, 
                    service_url, max_retries, retry_delay, request_timeout
                ) # Get the video clip paths

                current_image = [] # *  # No image to return for this tool
                current_video_paths = video_clip_paths  # List of video clip paths
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

                # save to clip_path_frame_"frame"_x_"coord_x"_y_"coord_y".png
                selected_image_path = f"{selected_clip_path}_frame_{frame_idx}.png"
                frame.save(selected_image_path)
                print(f"[DEBUG] tool ||{tool_name}|| saved to {selected_image_path}")

                info_need_to_stated = {
                    "tool_name": tool_name,
                    "clip_resolution": frame.size,
                    "selected_image_path": selected_image_path
                    }

            elif tool_name == "video_frame_grounder_tool": 
                # check whether the perceiver is called before
                if check_previous_tool_call(tool_name, kwargs['tools_call_state_list']): 
                    selected_image_path = get_ouput_from_previous_tool_call(tool_name, kwargs['tools_call_state_list'])
                else:
                    raise ValueError("[DEBUG] tool ||{tool_name}|| No selected image path available. Please run the video_image_perceiver_tool first.")

                bbox = args["bbox_2d"] # (x1,y1,x2,y2)
                img = Image.open(selected_image_path)
                # img_size = 

                cropped_img = img.crop(bbox)

                if cropped_img.size == (0,0):
                    raise ValueError("[DEBUG] tool ||{tool_name}|| [ERROR] crop into (0,0) image.")

                
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
            # video_str = "<video>"*len(current_video_paths) if current_video_paths else ""
            video_str = ", ".join(
                [f"video clip {i+1}: <video>" for i in range(len(current_video_paths))]
            ) if current_video_paths else ""
            # TODO DONE: support multi images/videos 
            obs = {
                "prompt": "<|im_end|>\n<|im_start|>user\n" + video_str + image_str + self.user_prompt + "<|im_end|>\n<|im_start|>assistant\n",
                # "prompt": "<|im_end|>\n<|im_start|>user\n" + video_str + image_str + self.user_prompt + "\nThink first, then call tools. Format strictly as: <think>...</think> <tool_call>...</tool_call>" + "<|im_end|>\n<|im_start|>assistant\n",

                "multi_modal_data": {"image": [current_image[0]] if current_image else [], "video": None}, #  current_image: List[Image]
                # "vision_infos": {"image": None, "video": [current_video_paths[0]] if current_video_paths else []},
                "vision_infos": {"image": None, "video": current_video_paths if current_video_paths else []},
                "tool_name": tool_name, 
            }
            reward = 0.5  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name, "info_stated": info_need_to_stated}
            # print(f'[DEBUG] SUCCESS ACTION AT STEP {len(kwargs["tools_call_state_list"])} The str is {action_string=}')
            return obs, reward, done, info
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)}')
            # print(f'[DEBUG] FAILED ACTION AT STEP {len(kwargs["tools_call_state_list"])} {action_string=}')
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
    tool = VideoToolBoxV5("video_toolbox", "Tool for video processing", {})
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