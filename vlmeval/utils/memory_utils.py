import gc
import torch
import psutil
import os
from typing import Optional


def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    gpu_memory = ""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB
        gpu_memory = f"GPU: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_cached:.2f}GB cached"
    
    return f"RAM: {memory_mb:.2f}MB, {gpu_memory}"


def force_memory_cleanup(verbose: bool = False):
    """强制清理内存"""
    if verbose:
        print(f"[MEMORY] Before cleanup: {get_memory_usage()}")
    
    # Python垃圾回收
    gc.collect()
    
    # GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 强制同步所有GPU操作
        torch.cuda.synchronize()
    
    # 再次垃圾回收
    gc.collect()
    
    if verbose:
        print(f"[MEMORY] After cleanup: {get_memory_usage()}")


def cleanup_variables(*variables, verbose: bool = False):
    """清理指定的变量"""
    if verbose:
        print(f"[MEMORY] Before variable cleanup: {get_memory_usage()}")
    
    for var in variables:
        if var is not None:
            del var
    
    force_memory_cleanup(verbose=False)
    
    if verbose:
        print(f"[MEMORY] After variable cleanup: {get_memory_usage()}")


class MemoryMonitor:
    """内存监控上下文管理器"""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_memory = None
    
    def __enter__(self):
        if self.verbose:
            self.start_memory = get_memory_usage()
            print(f"[MEMORY] {self.name} started: {self.start_memory}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        force_memory_cleanup(verbose=False)
        if self.verbose:
            end_memory = get_memory_usage()
            print(f"[MEMORY] {self.name} finished: {end_memory}")


def monitor_memory_growth(threshold_mb: float = 100.0, action: str = "warn"):
    """监控内存增长，超过阈值时采取行动"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            before_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            after_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_increase = after_memory - before_memory
            
            if memory_increase > threshold_mb:
                print(f"[MEMORY WARNING] Function {func.__name__} increased memory by {memory_increase:.2f}MB")
                if action == "cleanup":
                    force_memory_cleanup(verbose=True)
                elif action == "raise":
                    raise RuntimeError(f"Memory increase {memory_increase:.2f}MB exceeds threshold {threshold_mb}MB")
            
            return result
        return wrapper
    return decorator
