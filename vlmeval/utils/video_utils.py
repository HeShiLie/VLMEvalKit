"""
通用视频帧转换工具，解决不同版本decord和torch的兼容性问题
"""
import numpy as np


def safe_frame_to_numpy(frame):
    """
    安全地将视频帧转换为numpy数组，兼容不同版本的decord和torch
    
    Args:
        frame: 视频帧对象，可能是decord.NDArray、torch.Tensor或其他类型
        
    Returns:
        numpy.ndarray: 转换后的numpy数组
    """
    try:
        # 尝试使用asnumpy()方法 (decord默认方法)
        return frame.asnumpy()
    except AttributeError:
        # 如果没有asnumpy()方法，尝试其他转换方式
        try:
            # 尝试使用numpy()方法 (PyTorch Tensor方法)
            return frame.numpy()
        except AttributeError:
            # 如果都没有，尝试直接转换
            try:
                import torch
                if torch.is_tensor(frame):
                    return frame.detach().cpu().numpy()
                else:
                    return np.array(frame)
            except:
                # 最后的回退方案
                return np.array(frame)


def safe_batch_to_numpy(batch_frames):
    """
    安全地将批量视频帧转换为numpy数组
    
    Args:
        batch_frames: 批量视频帧对象
        
    Returns:
        numpy.ndarray: 转换后的numpy数组
    """
    try:
        # 尝试使用asnumpy()方法
        return batch_frames.asnumpy()
    except AttributeError:
        # 如果没有asnumpy()方法，尝试其他转换方式
        try:
            # 尝试使用numpy()方法
            return batch_frames.numpy()
        except AttributeError:
            # 如果都没有，尝试直接转换
            try:
                import torch
                if torch.is_tensor(batch_frames):
                    return batch_frames.detach().cpu().numpy()
                else:
                    return np.array(batch_frames)
            except:
                # 最后的回退方案
                return np.array(batch_frames)
