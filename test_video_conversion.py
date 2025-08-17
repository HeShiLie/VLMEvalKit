#!/usr/bin/env python3
"""
测试脚本：验证视频帧转换的兼容性修复
"""

import numpy as np
import torch
from vlmeval.utils.video_utils import safe_frame_to_numpy, safe_batch_to_numpy

def test_frame_conversion():
    """测试不同类型的视频帧转换"""
    
    print("Testing frame conversion compatibility...")
    
    # 测试1: numpy数组
    np_frame = np.random.rand(224, 224, 3) * 255
    np_frame = np_frame.astype(np.uint8)
    result1 = safe_frame_to_numpy(np_frame)
    print(f"✅ NumPy array: {result1.shape}, {result1.dtype}")
    
    # 测试2: PyTorch张量
    torch_frame = torch.rand(224, 224, 3) * 255
    torch_frame = torch_frame.to(torch.uint8)
    result2 = safe_frame_to_numpy(torch_frame)
    print(f"✅ PyTorch tensor: {result2.shape}, {result2.dtype}")
    
    # 测试3: 模拟decord对象（创建一个有asnumpy方法的类）
    class MockDecordFrame:
        def __init__(self, data):
            self.data = data
        
        def asnumpy(self):
            return self.data
    
    mock_frame = MockDecordFrame(np_frame)
    result3 = safe_frame_to_numpy(mock_frame)
    print(f"✅ Mock decord frame: {result3.shape}, {result3.dtype}")
    
    # 测试4: 模拟没有asnumpy但有numpy方法的对象
    class MockTensorFrame:
        def __init__(self, data):
            self.data = data
        
        def numpy(self):
            return self.data
    
    mock_tensor = MockTensorFrame(np_frame)
    result4 = safe_frame_to_numpy(mock_tensor)
    print(f"✅ Mock tensor frame: {result4.shape}, {result4.dtype}")
    
    print("\n🎉 All frame conversion tests passed!")

if __name__ == "__main__":
    test_frame_conversion()
