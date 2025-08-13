# VLMEvalKit 并行评测功能

## 概述

本次更新为 VLMEvalKit 添加了并行评测功能，可以显著提升评测效率，特别是在使用 vLLM 进行推理时。主要改进包括：

1. **数据集级并行处理**：多个数据集可以并行处理，而不是串行执行
2. **批量推理支持**：为支持的模型（如 VideoRLQwen）添加了批量推理功能
3. **灵活的并行控制**：可以通过命令行参数控制并行度和批量大小

## 主要特性

### 1. 数据集级并行处理
- 使用 `ThreadPoolExecutor` 并行处理多个数据集
- 可通过 `--max-workers` 参数控制并行度
- 默认并行度为 `min(数据集数量, CPU核心数)`

### 2. 批量推理优化
- 为 VideoRLQwen 模型添加了 `generate_batch` 方法
- 支持 vLLM 的批量推理能力
- 可通过 `--batch-size` 参数控制批量大小

### 3. 向后兼容
- 保留了原有的串行处理逻辑
- 可通过 `--no-parallel` 参数禁用并行处理
- 不影响现有的评测脚本

## 使用方法

### 基本用法

```bash
# 启用并行处理（默认）
python run.py --config your_config.json --parallel --max-workers 4 --batch-size 8

# 禁用并行处理
python run.py --config your_config.json --no-parallel

# 使用命令行参数而非配置文件
python run.py --model VideoRLQwen --data Video-MME_8frame MVBench_8frame --parallel
```

### 新增命令行参数

- `--parallel` / `--no-parallel`: 启用/禁用并行处理（默认启用）
- `--max-workers`: 最大并行工作线程数（默认：min(数据集数量, CPU核心数)）
- `--batch-size`: vLLM 批量推理的批量大小（默认：8）

### 配置文件示例

```json
{
    "model": {
        "VideoRLQwen_Parallel": {
            "class": "VideoRLQwen",
            "model_path": "/path/to/your/model",
            "use_vllm": true
        }
    },
    "data": {
        "Video-MME_8frame": {
            "class": "VideoMME",
            "dataset": "Video-MME",
            "nframe": 8
        },
        "MVBench_8frame": {
            "class": "MVBench",
            "dataset": "MVBench",
            "nframe": 8
        },
        "Video_Holmes_32frame": {
            "class": "Video_Holmes",
            "dataset": "Video_Holmes",
            "nframe": 32
        }
    }
}
```

## 性能优化建议

### 1. 并行度设置
- **CPU密集型任务**：设置 `max-workers` 为 CPU 核心数
- **GPU密集型任务**：根据 GPU 内存限制调整，通常 2-4 个工作线程
- **混合任务**：从较小的值开始测试，逐步增加

### 2. 批量大小设置
- **大显存GPU**：可以设置较大的 batch-size（16-32）
- **中等显存GPU**：推荐 batch-size 为 8-16
- **小显存GPU**：建议 batch-size 为 4-8

### 3. 内存管理
- 并行处理会增加内存使用
- 监控系统内存和GPU内存使用情况
- 必要时减少 `max-workers` 或 `batch-size`

## 代码结构

### 新增文件
- `vlmeval/inference_video_parallel.py`: 并行推理实现
- `parallel_config_example.json`: 配置文件示例
- `run_parallel_example.sh`: 运行脚本示例

### 修改文件
- `run.py`: 添加并行处理逻辑和新的命令行参数
- `vlmeval/vlm/video_tool_rl_gao/video_tool_rl.py`: 为 VideoRLQwen 添加批量推理方法

## 注意事项

### 1. 线程安全
- 确保模型实例是线程安全的
- 避免在并行线程间共享可变状态

### 2. GPU内存管理
- 并行处理可能增加GPU内存使用
- 使用 `torch.cuda.empty_cache()` 及时释放内存

### 3. 错误处理
- 单个数据集失败不会影响其他数据集的处理
- 所有错误都会被记录到日志中

### 4. 分布式训练兼容性
- 并行功能与现有的分布式训练兼容
- 在分布式环境中，每个进程内部仍可使用并行处理

## 故障排除

### 1. 内存不足
```
解决方案：
- 减少 --max-workers 参数
- 减少 --batch-size 参数
- 使用 --no-parallel 回退到串行处理
```

### 2. GPU内存不足
```
解决方案：
- 减少 --batch-size 参数
- 确保正确调用 torch.cuda.empty_cache()
- 检查GPU内存碎片问题
```

### 3. 并行性能不佳
```
解决方案：
- 检查是否存在GIL（Global Interpreter Lock）限制
- 考虑使用 ProcessPoolExecutor 而非 ThreadPoolExecutor
- 分析瓶颈是否在I/O或计算
```

## 示例运行

```bash
# 查看帮助
python run.py --help

# 运行示例配置
bash run_parallel_example.sh

# 自定义并行配置
python run.py \
    --config your_config.json \
    --parallel \
    --max-workers 3 \
    --batch-size 16 \
    --verbose \
    --log-file evaluation.log
```

## 性能对比

使用并行处理后，预期性能提升：

- **数据集级并行**：3-4个数据集并行处理可提升 200%-300% 的吞吐量
- **批量推理**：批量大小为8时，推理速度可提升 150%-200%
- **总体提升**：在理想情况下，总体评测时间可缩短 60%-80%

*实际性能提升取决于硬件配置、数据集大小和模型复杂度*
