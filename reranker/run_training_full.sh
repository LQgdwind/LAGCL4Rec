#!/bin/bash

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -c "import torch; torch.cuda.empty_cache()"

mkdir -p ./output/qwen_reranker_full

LR=1e-5

MAX_POS_ITEMS=10   # 每个用户的最大正样本数
MAX_CANDIDATES=20  # 每个正样本的最大候选项数

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 ${NUM_GPUS} GPUs"

deepspeed --include=localhost:0,1,2,3,4,5,6 qwen_reranker_trainer.py \
    --model_name_or_path ./downloaded_models/Qwen2.5-7B-Instruct \
    --profile_path ../dataset/ml-1m/merged_profiles.json \
    --test_file_path ../dataset/ml-1m/test.txt \
    --cf_candidates_path ../dataset/ml-1m/cf_candidates.json \
    --output_dir ./output/qwen_reranker_full \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate ${LR} \
    --max_pos_items ${MAX_POS_ITEMS} \
    --max_candidates ${MAX_CANDIDATES} \
    --deepspeed_config ./ds_config_full.json \
    --weight_decay 0.01

echo "Training completed!" 