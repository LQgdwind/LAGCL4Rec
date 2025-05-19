import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    default_data_collator,
    HfArgumentParser,
)
from transformers.trainer_utils import set_seed
import deepspeed
import random
import gc  # 添加垃圾回收模块
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 设置基本日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# 固定随机种子以确保可重复性
set_seed(42)

def load_profiles(profile_path):
    """加载用户和物品的个人资料"""
    logger.info(f"Loading profiles from {profile_path}...")
    with open(profile_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    
    # 构建查找字典
    user_profiles = {}
    for entry in profiles:
        user_id = entry.get('user_id')
        if not user_id:
            continue
        
        # 存储用户档案
        user_profiles[user_id] = {
            'user_profile': entry.get('user_profile', {}),
            'positive_items': {},
            'negative_items': {}
        }
        
        # 存储正面项目
        for item in entry.get('positive_items', []):
            item_id = item.get('item_id')
            if item_id:
                user_profiles[user_id]['positive_items'][item_id] = item.get('item_profile', {})
        
        # 存储负面项目
        for item in entry.get('negative_items', []):
            item_id = item.get('item_id')
            if item_id:
                user_profiles[user_id]['negative_items'][item_id] = item.get('item_profile', {})
    
    logger.info(f"Loaded profiles for {len(user_profiles)} users")
    return user_profiles

def load_test_data(test_file_path):
    """加载测试数据作为标签/真实交互"""
    user_items = {}
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = parts[0]
                item_id = parts[1]
                if user_id not in user_items:
                    user_items[user_id] = []
                user_items[user_id].append(item_id)
    print(f"从{test_file_path}加载了{len(user_items)}个用户的测试数据")
    return user_items

def load_cf_candidates(cf_candidates_path):
    """加载协同过滤模型生成的候选项"""
    with open(cf_candidates_path, 'r', encoding='utf-8') as f:
        cf_candidates = json.load(f)
    print(f"从{cf_candidates_path}加载了{len(cf_candidates)}个用户的协同过滤候选项")
    return cf_candidates

def get_item_details(item_id, user_profiles, user_id):
    """获取物品的详细信息"""
    if user_id not in user_profiles:
        return f"Item {item_id}"
    
    # 检查物品是否在用户喜欢或不喜欢的项目中
    positive_items = user_profiles[user_id].get('positive_items', {})
    negative_items = user_profiles[user_id].get('negative_items', {})
    
    if item_id in positive_items:
        item_profile = positive_items[item_id]
        status = "(User previously liked this)"
    elif item_id in negative_items:
        item_profile = negative_items[item_id]
        status = "(User previously disliked this)"
    else:
        item_profile = {}
        status = ""
    
    # 获取物品描述
    summarization = item_profile.get('summarization', f"Item {item_id}")
    return f"Item {item_id} {status}: {summarization}"

def construct_reranker_prompt(user_id, item_id, user_profiles):
    """
    构建训练阶段的提示模板，与推理阶段完全保持一致，包括思维链(Chain of Thought)
    
    Args:
        user_id: 用户ID
        item_id: 物品ID
        user_profiles: 用户资料字典
    
    Returns:
        为单个用户-物品对构建的提示，与推理阶段格式相同
    """
    # 创建提示
    prompt = "You are a Precision Reranking Engine.\n"
    prompt += "Your sole task is to reorder the provided list of candidates strictly and exclusively based on the user preferences detailed below.\n\n"
    
    # 获取用户资料
    prompt += "USER INFORMATION\n"
    if user_id in user_profiles:
        profile_data = user_profiles[user_id].get('user_profile', {})
        summarization = profile_data.get('summarization', '')
        if summarization:
            prompt += f"User Information: User {user_id} with explicit preferences: {summarization}\n\n"
        else:
            prompt += f"User Information: User {user_id}\n\n"
    else:
        prompt += f"User Information: User {user_id}\n\n"
    
    # 添加候选项
    prompt += "CANDIDATE ITEMS\n"
    prompt += f"Candidate Items List: {item_id}\n"
    prompt += f"Original Ranking: {item_id}\n\n"
    
    # 添加用户交互历史
    prompt += "USER INTERACTION HISTORY\n"
    if user_id in user_profiles:
        # 添加历史信息 - 用户喜欢的物品
        positive_items = user_profiles[user_id].get('positive_items', {})
        if positive_items:
            prompt += "Items the user previously liked (strong relevance signals):\n"
            for i, (pos_id, profile) in enumerate(list(positive_items.items())[:5]):
                if pos_id != item_id:  # 避免重复当前物品
                    summarization = profile.get('summarization', '')
                    if summarization:
                        prompt += f"Item {pos_id}: {summarization}\n"
            prompt += "\n"
        
        # 添加历史信息 - 用户不喜欢的物品
        negative_items = user_profiles[user_id].get('negative_items', {})
        if negative_items:
            prompt += "Items the user previously disliked (negative signals):\n"
            for i, (neg_id, profile) in enumerate(list(negative_items.items())[:3]):
                if neg_id != item_id:  # 避免重复当前物品
                    summarization = profile.get('summarization', '')
                    if summarization:
                        prompt += f"Item {neg_id}: {summarization}\n"
            prompt += "\n"
    
    # 添加候选物品详情
    prompt += "CANDIDATE ITEM DETAILS\n"
    
    # 获取物品资料
    if user_id in user_profiles:
        positive_items = user_profiles[user_id].get('positive_items', {})
        negative_items = user_profiles[user_id].get('negative_items', {})
        
        item_signal = ""
        item_profile = {}
        
        if item_id in positive_items:
            item_profile = positive_items[item_id]
            item_signal = "(User previously liked similar items)"
        elif item_id in negative_items:
            item_profile = negative_items[item_id]
            item_signal = "(User previously disliked similar items)"
        
        summarization = item_profile.get('summarization', f"Item {item_id}")
        prompt += f"Item {item_id} {item_signal}: {summarization}\n\n"
    else:
        prompt += f"Item {item_id}\n\n"
    
    # 添加排序推理过程
    prompt += "RANKING REASONING PROCESS\n"
    prompt += "Follow this reasoning process:\n"
    prompt += "Step 1: Prioritize items that closely match the user's historical preferences and explicit likes.\n"
    prompt += "Step 2: Demote items similar to those the user has historically disliked.\n"
    prompt += "Step 3: Make minimal adjustments to the original ranking, changing positions only when you have high confidence.\n\n"
    
    # 添加排序指南
    prompt += "RANKING GUIDELINES\n"
    prompt += "1. Promote the most relevant items to top positions\n"
    prompt += "2. Make conservative adjustments - only move items when necessary\n"
    prompt += "3. The original ranking is usually correct - only adjust with clear evidence\n\n"
    
    # 训练阶段特定指导
    prompt += "For this single-item evaluation:\n"
    prompt += "- If this item should be ranked highly based on user preferences, classify as relevant (1)\n"
    prompt += "- If this item should be ranked lower, classify as less relevant (0)\n"
    
    return prompt

def create_training_examples_batch(user_ids, profiles, test_data, cf_candidates, max_pos_items=3, max_candidates=20, batch_size=100):
    """
    批量创建训练样本，以减少内存使用
    """
    examples = []
    total_users = len(user_ids)
    
    for i in range(0, total_users, batch_size):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_examples = []
        
        for user_id in batch_user_ids:
            positive_items = test_data.get(user_id, [])
            candidates = cf_candidates.get(user_id, [])
            
            if not positive_items or not candidates:
                continue
                
            # 为了减少内存使用，限制处理的正样本数量
            for pos_item_id in positive_items[:max_pos_items]:  # 使用参数控制正样本数量
                # 确保候选项列表不会太长，这可能导致内存问题
                current_candidates = candidates[:max_candidates]  # 使用参数控制候选项数量
                
                # 确保正样本在候选项中
                if pos_item_id not in current_candidates:
                    current_candidates = [pos_item_id] + current_candidates[:max_candidates-1]
                else:
                    current_candidates.remove(pos_item_id)
                    current_candidates = [pos_item_id] + current_candidates
                
                # 为每个物品创建训练样本，使用与llm_reranker.py相同的提示模板
                for item_id in current_candidates:
                    # 使用与推理时相同的提示模板
                    input_text = construct_reranker_prompt(user_id, item_id, profiles)
                    
                    # 正样本的标签为1，负样本的标签为0
                    label = 1 if item_id == pos_item_id else 0
                    
                    batch_examples.append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "input_text": input_text,
                        "label": label,
                    })
        
        examples.extend(batch_examples)
        # 手动触发垃圾回收，释放内存
        batch_examples.clear()
        gc.collect()
        logger.info(f"已处理 {min(i + batch_size, total_users)}/{total_users} 用户，当前样本数: {len(examples)}")
    
    logger.info(f"创建了{len(examples)}个训练样本")
    return examples

def tokenize_function(examples, tokenizer):
    """对训练样本进行标记化处理，使用单一输入文本字段"""
    inputs = tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # 减少最大长度以降低内存使用
        return_tensors="pt"
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": examples["label"]
    }

def create_training_examples(profiles, test_data, cf_candidates, num_examples=None, max_pos_items=3, max_candidates=20):
    """
    创建训练样本
    - profiles: 用户档案
    - test_data: 真实交互数据（标签）
    - cf_candidates: 协同过滤模型生成的候选项
    - num_examples: 要使用的示例数量（用户数量）
    - max_pos_items: 每个用户最多使用的正样本数量
    - max_candidates: 每个正样本最多使用的候选项数量
    """
    # 找出同时存在于测试数据和协同过滤候选项中的用户
    common_users = set(test_data.keys()) & set(cf_candidates.keys())
    logger.info(f"在测试数据和协同过滤候选项中找到{len(common_users)}个共同用户")
    
    # 对用户ID列表进行随机排序
    user_ids = list(common_users)
    random.shuffle(user_ids)
    
    # 如果指定了样本数，则限制处理的用户数
    if num_examples is not None:
        max_users = min(num_examples, len(user_ids))
        user_ids = user_ids[:max_users]
    
    # 使用批量处理方式创建训练样本
    return create_training_examples_batch(user_ids, profiles, test_data, cf_candidates, 
                                          max_pos_items=max_pos_items, 
                                          max_candidates=max_candidates, 
                                          batch_size=50)

def train_model(args):
    """训练模型主函数"""
    # 加载数据
    logger.info("开始加载数据...")
    user_profiles = load_profiles(args.profile_path)
    cf_candidates = load_cf_candidates(args.cf_candidates_path)
    test_data = load_test_data(args.test_file_path)
    
    # 创建训练示例
    logger.info("创建训练示例...")
    examples = create_training_examples(user_profiles, test_data, cf_candidates, 
                                      args.num_examples,
                                      max_pos_items=args.max_pos_items,
                                      max_candidates=args.max_candidates)
    
    # 释放不再需要的数据
    del user_profiles
    del cf_candidates
    del test_data
    gc.collect()
    
    # 加载tokenizer和模型
    logger.info(f"Loading Qwen2.5-7B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # 配置特殊token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载模型 - 默认使用全参数微调
    logger.info("Loading model for full parameter fine-tuning...")
    
    # 清理缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 获取当前GPU索引
    local_rank = args.local_rank if args.local_rank != -1 else 0
    
    # 显式指定设备映射到当前设备
    device_map = {'': local_rank}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        device_map=device_map,  # 使用正确的设备映射
        use_cache=False,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True
    )
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of {total_params} total parameters)")
    
    # 准备数据集
    logger.info("准备数据集...")
    
    # 减小Dataset的批次大小，以降低内存使用
    chunk_size = 2000  # 减少每批次处理的示例数，以降低内存使用
    tokenized_examples = []  # 存储所有标记化后的示例
    
    for i in range(0, len(examples), chunk_size):
        chunk = examples[i:i+chunk_size]
        chunk_dataset = Dataset.from_dict({
            "user_id": [ex["user_id"] for ex in chunk],
            "item_id": [ex["item_id"] for ex in chunk],
            "input_text": [ex["input_text"] for ex in chunk],
            "label": [ex["label"] for ex in chunk]
        })
        
        # 对数据集进行标记化处理
        tokenized_chunk = chunk_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            batch_size=64,  # 减小处理批次大小
            remove_columns=chunk_dataset.column_names
        )
        
        # 将tokenized_chunk转换为列表并添加到tokenized_examples
        tokenized_examples.extend(tokenized_chunk)
        
        # 手动清理内存
        del chunk
        del chunk_dataset
        del tokenized_chunk
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"已处理 {min(i + chunk_size, len(examples))}/{len(examples)} 个训练示例")
    
    # 从列表创建一个新的Dataset
    tokenized_dataset = Dataset.from_list(tokenized_examples)
    
    # 清理不再需要的数据
    del examples
    del tokenized_examples
    torch.cuda.empty_cache()
    gc.collect()
    
    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查看可用GPU
    n_gpus = torch.cuda.device_count()
    logger.info(f"可用GPU数量: {n_gpus}")
    
    # 检查CUDA设备当前内存使用情况
    for i in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        logger.info(f"GPU {i} ({gpu_name}) - 已分配内存: {mem_allocated:.2f} GB, 已保留内存: {mem_reserved:.2f} GB")
    
    # 训练参数 - 针对全参数微调进行优化
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,  # 极小批量大小以避免OOM
        gradient_accumulation_steps=8,  # 减少梯度累积步数以加快训练
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.03,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,  # 减少保存的检查点数量
        fp16=False,
        bf16=True,
        optim="adamw_torch",  # 使用普通优化器替代fused版本避免兼容性问题
        remove_unused_columns=False,
        report_to="none",
        deepspeed=None,  # 禁用DeepSpeed以解决'ds_grads_remaining'属性错误
        local_rank=args.local_rank,
        gradient_checkpointing=True,
        dataloader_num_workers=1,  # 使用单个工作线程加载数据
        dataloader_pin_memory=False,  # 禁用pin_memory以减少GPU内存使用
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,  # 添加梯度裁剪以提高训练稳定性
        group_by_length=True,  # 按长度分组，可能提高训练效率
        seed=42,
    )
    
    # 创建Trainer，指定特定的数据整理器以减少内存使用
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )
    
    # 训练模型
    logger.info("Starting training...")
    trainer.train()
    
    # 保存模型
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

@dataclass
class RerankerTrainingArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train_file: str = field(
        default=None, metadata={"help": "The input training data file (a json file)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input validation data file (a json file)."}
    )
    output_dir: str = field(
        default="output/qwen_reranker",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW optimizer."}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    save_total_limit: Optional[int] = field(
        default=5,
        metadata={"help": "Limit the total amount of checkpoints, delete the older checkpoints."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization."}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Enable deepspeed and pass the path to deepspeed json config file."}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training on GPUs."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing to save memory at the expense of slower backward pass."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run an evaluation every X steps."}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether or not to load the best model found during training at the end"}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )

def main():
    parser = HfArgumentParser(RerankerTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    set_seed(args.seed)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = RerankerDataset(tokenizer, args.train_file, args.max_length)
    
    eval_dataset = None
    if args.validation_file:
        eval_dataset = RerankerDataset(tokenizer, args.validation_file, args.max_length)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.validation_file:
        metrics = trainer.evaluate()
        print(f"Validation metrics: {metrics}")

if __name__ == "__main__":
    main()