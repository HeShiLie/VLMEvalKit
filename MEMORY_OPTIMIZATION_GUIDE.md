# Video-MME 内存优化方案

## 问题分析

Video-MME 运行时内存不断增长的主要原因：

1. **视频解码器未释放**: `decord.VideoReader` 对象在使用后没有被明确释放
2. **视频数据累积**: 视频张量和相关数据在处理后没有及时清理
3. **批量数据残留**: 批量推理过程中的中间数据没有清理
4. **GPU缓存未清理**: CUDA缓存在每次推理后没有清空
5. **Python对象引用**: 大量的中间变量没有被垃圾回收

## 优化措施

### 1. VideoMME 数据集级别优化

**位置**: `vlmeval/dataset/videomme.py`

- 在 `save_video_frames()` 函数中显式删除 `decord.VideoReader` 对象
- 清理解码后的图像数据
- 添加垃圾回收调用

```python
# 明确释放video reader对象
del vid
import gc
gc.collect()
```

### 2. VideoRLQwen 模型级别优化

**位置**: `vlmeval/vlm/video_tool_rl_gao/video_tool_rl.py`

#### 批量推理优化:
- 使用内存监控器跟踪内存使用
- 批量处理完成后清理所有中间变量
- 强制GPU缓存清理

#### 单个推理优化:
- 在 `generate_inner()` 中清理视频数据
- 删除处理过的张量和变量

### 3. 推理引擎级别优化

**位置**: `vlmeval/inference_video_parallel.py`

- 每个批次处理后清理批量数据
- 串行处理中每个样本后清理结构体
- 增加垃圾回收频率

### 4. 新增内存管理工具

**位置**: `vlmeval/utils/memory_utils.py`

提供了专门的内存管理工具:

- `get_memory_usage()`: 监控内存使用情况
- `force_memory_cleanup()`: 强制内存清理
- `MemoryMonitor`: 上下文管理器监控内存变化
- `monitor_memory_growth()`: 装饰器监控函数内存增长

### 5. 环境变量修复

**位置**: `vlmeval/vlm/video_tool_rl_gao/video_tools/parallel_envs.py`

- 修复了 `os.environ['RANK']` 的安全访问问题

## 使用建议

### 1. 运行时监控
```bash
# 在运行过程中监控内存使用
watch -n 1 'nvidia-smi && free -h'
```

### 2. 批次大小调整
根据可用内存调整批次大小:
```bash
# 对于8GB GPU，建议批次大小为4-8
--batch-size 4

# 对于16GB GPU，建议批次大小为8-16  
--batch-size 8
```

### 3. 采样数量控制
使用采样参数减少总体内存需求:
```bash
# 先用小样本测试
--sample-size 100
```

### 4. 监控内存增长
代码中已集成内存监控，运行时会自动输出内存使用情况。

## 预期效果

1. **内存泄漏消除**: 视频处理后内存会被正确释放
2. **内存使用稳定**: 长时间运行内存不会持续增长
3. **GPU内存优化**: CUDA缓存会被及时清理
4. **性能提升**: 减少内存压力，提高处理速度

## 验证方法

1. **内存监控**: 观察运行过程中内存使用是否稳定
2. **长时间测试**: 运行大量样本查看内存是否持续增长
3. **GPU监控**: 检查GPU内存是否在合理范围内

## 注意事项

1. 这些优化可能会略微增加垃圾回收的开销
2. 内存监控输出可能会增加日志量
3. 如果遇到问题，可以通过设置 `verbose=False` 减少输出
