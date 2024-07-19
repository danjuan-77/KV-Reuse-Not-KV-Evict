#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server \
    --model ./phi3_mini \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --api-key token-abc123 \
    --gpu_memory_utilization 0.2