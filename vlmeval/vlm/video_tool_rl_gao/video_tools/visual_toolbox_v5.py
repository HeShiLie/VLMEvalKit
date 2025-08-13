import numpy as np
import copy
try:
    from .tool_envs import ToolBase
except ImportError:
    # For standalone testing
    from tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
import traceback
from math import ceil, floor

class VisualToolBoxV5(ToolBase):
    name = "visual_toolbox_v5"
    user_prompt = "Here is the observation returned after you executed the tool call."
    max_action_per_turn = 3
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return [{"tool_call": call.strip()} for call in tool_calls] if tool_calls else []

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
        
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}
        # try:
        #     tool_call = json.loads(action.strip())  # 或使用 literal_eval
        # except Exception as e:
        #     error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
        #     obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n"
        #     info = {"error": str(e), "status": "failed"}
        #     return obs, 0.0, False, {}
        if len(action) > self.max_action_per_turn:
            action = action[:self.max_action_per_turn]
        current_image = []
        tool_call_str = "\n"
        try:
            for act in action:
                tool_call_str += act['tool_call'] 
                tool_call_str += "\n"
                tool_call = json.loads(act['tool_call'])
                tool_name = tool_call["name"]
                args = tool_call["arguments"]

                if tool_name == "image_zoom_in_tool":
                    # Zoom in by cropping the image
                    bbox = args["bbox_2d"]
                    bbox = self.maybe_resize_bbox(*bbox)
                    if not bbox:
                        raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                    img = self.multi_modal_data['image'][0]
                    cropped_img = img.crop(bbox)
                    current_image.append(cropped_img)
                    
                elif tool_name == "image_rotate_tool":
                    # Rotate the image
                    angle = args["angle"]
                    img = self.multi_modal_data['image'][0]
                    rotated_img = img.rotate(angle)
                    current_image.append(rotated_img)
                    
                else:
                    raise ValueError(f"Unknown tool name: {tool_name}")
                    
            # Prepare the observation
            image_str = "<image>" if current_image else ""
            obs = {
                "prompt": "<|im_end|>\n<|im_start|>user\n" + image_str + self.user_prompt + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": current_image if current_image else [], "video": None},
                "vision_infos": {"image": None, "video": []},
                "tool_name": tool_name, # TODO: delete after debug
            }
            reward = 0.5  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG] SUCCESS ACTION AT STEP {len(kwargs.get("tools_call_state_list", []))} The str is {action_string=}')
            return obs, reward, done, info
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)}')
            print(f'[DEBUG] FAILED ACTION AT STEP {len(kwargs.get("tools_call_state_list", []))} {action_string=}')
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
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False


    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]
    
if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualToolBoxV5("visual_toolbox", "Tool for image processing", {})
    
    # Setup the tool with test data
    dummy_image = Image.new('RGB', (200, 200), color='white')
    tool.multi_modal_data = {"image": [dummy_image]}
    tool.reset("test", {"image": [dummy_image]}, {"image": [dummy_image]})
    
    # Test zoom in tool (should return reward=0.5)
    zoom_in_action = """
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 10, 100, 100]}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    
    # Test rotate tool (should return reward=0.5)
    rotate_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"angle": 90}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(rotate_action)
    print(f"Rotate result - Reward: {reward}, Info: {info}")
    
    # Test invalid JSON (should return reward=0.0)
    invalid_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"angle": 90}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(invalid_action)
    print(f"Valid JSON result - Reward: {reward}, Info: {info}")
    
    # Test unknown tool (should return reward=0.0)
    unknown_tool_action = """
    <tool_call>
    {"name": "unknown_tool", "arguments": {"param": "value"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(unknown_tool_action)
    print(f"Unknown tool result - Reward: {reward}, Info: {info}")
    
    # Test answer extraction
    answer_action = """
    <answer>This is my final answer</answer>
    """
    obs, reward, done, info = tool.execute(answer_action)
    print(f"Answer result - Reward: {reward}, Done: {done}, Info: {info}")