#!/bin/bash

# 并行评测示例脚本
# 使用配置文件进行并行评测

echo "开始并行评测..."

# 方式1: 使用配置文件进行并行评测
python run.py \
    --config parallel_config_example.json \
    --work-dir ./outputs \
    --use-vllm \
    --parallel \
    --max-workers 3 \
    --batch-size 8 \
    --verbose \
    --log-file ./parallel_evaluation.log

echo "并行评测完成！"

# 方式2: 使用命令行参数进行并行评测（如果不使用配置文件）
# python run.py \
#     --model VideoRLQwen \
#     --data Video-MME_8frame MVBench_8frame Video_Holmes_32frame \
#     --work-dir ./outputs \
#     --use-vllm \
#     --parallel \
#     --max-workers 3 \
#     --batch-size 8 \
#     --verbose \
#     --log-file ./parallel_evaluation.log

# 方式3: 禁用并行，使用串行评测
# python run.py \
#     --config parallel_config_example.json \
#     --work-dir ./outputs \
#     --use-vllm \
#     --no-parallel \
#     --batch-size 8 \
#     --verbose \
#     --log-file ./serial_evaluation.log
