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
import gc  
from dataclasses import dataclass, field
from typing import Dict, List, Optional


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


set_seed(42)

def load_profiles(profile_path):
    logger.info(f"Loading profiles from {profile_path}...")
    with open(profile_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    
    
    user_profiles = {}
    for entry in profiles:
        user_id = entry.get('user_id')
        if not user_id:
            continue
        
        user_profiles[user_id] = {
            'user_profile': entry.get('user_profile', {}),
            'positive_items': {},
            'negative_items': {}
        }
        
        for item in entry.get('positive_items', []):
            item_id = item.get('item_id')
            if item_id:
                user_profiles[user_id]['positive_items'][item_id] = item.get('item_profile', {})
        
        for item in entry.get('negative_items', []):
            item_id = item.get('item_id')
            if item_id:
                user_profiles[user_id]['negative_items'][item_id] = item.get('item_profile', {})
    
    logger.info(f"Loaded profiles for {len(user_profiles)} users")
    return user_profiles

def load_test_data(test_file_path):
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
    with open(cf_candidates_path, 'r', encoding='utf-8') as f:
        cf_candidates = json.load(f)
    print(f"从{cf_candidates_path}加载了{len(cf_candidates)}个用户的协同过滤候选项")
    return cf_candidates

def get_item_details(item_id, user_profiles, user_id):
    if user_id not in user_profiles:
        return f"Item {item_id}"
    
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
    
    summarization = item_profile.get('summarization', f"Item {item_id}")
    return f"Item {item_id} {status}: {summarization}"

def construct_reranker_prompt(user_id, item_id, user_profiles):
    prompt = "You are a Precision Reranking Engine.\n"
    prompt += "Your sole task is to reorder the provided list of candidates strictly and exclusively based on the user preferences detailed below.\n\n"
    
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
    
    prompt += "CANDIDATE ITEMS\n"
    prompt += f"Candidate Items List: {item_id}\n"
    prompt += f"Original Ranking: {item_id}\n\n"
    
    prompt += "USER INTERACTION HISTORY\n"
    if user_id in user_profiles:
        positive_items = user_profiles[user_id].get('positive_items', {})
        if positive_items:
            prompt += "Items the user previously liked (strong relevance signals):\n"
            for i, (pos_id, profile) in enumerate(list(positive_items.items())[:5]):
                if pos_id != item_id:
                    summarization = profile.get('summarization', '')
                    if summarization:
                        prompt += f"Item {pos_id}: {summarization}\n"
            prompt += "\n"
        
        negative_items = user_profiles[user_id].get('negative_items', {})
        if negative_items:
            prompt += "Items the user previously disliked (negative signals):\n"
            for i, (neg_id, profile) in enumerate(list(negative_items.items())[:3]):
                if neg_id != item_id:
                    summarization = profile.get('summarization', '')
                    if summarization:
                        prompt += f"Item {neg_id}: {summarization}\n"
            prompt += "\n"
    
    prompt += "CANDIDATE ITEM DETAILS\n"
    
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
    
    prompt += "RANKING REASONING PROCESS\n"
    prompt += "Follow this reasoning process:\n"
    prompt += "Step 1: Prioritize items that closely match the user's historical preferences and explicit likes.\n"
    prompt += "Step 2: Demote items similar to those the user has historically disliked.\n"
    prompt += "Step 3: Make minimal adjustments to the original ranking, changing positions only when you have high confidence.\n\n"
    
    prompt += "RANKING GUIDELINES\n"
    prompt += "1. Promote the most relevant items to top positions\n"
    prompt += "2. Make conservative adjustments - only move items when necessary\n"
    prompt += "3. The original ranking is usually correct - only adjust with clear evidence\n\n"
    
    prompt += "For this single-item evaluation:\n"
    prompt += "- If this item should be ranked highly based on user preferences, classify as relevant (1)\n"
    prompt += "- If this item should be ranked lower, classify as less relevant (0)\n"
    
    return prompt

def create_training_examples_batch(user_ids, profiles, test_data, cf_candidates, max_pos_items=3, max_candidates=20, batch_size=100):
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
                
            for pos_item_id in positive_items[:max_pos_items]:
                current_candidates = candidates[:max_candidates]
                
                if pos_item_id not in current_candidates:
                    current_candidates = [pos_item_id] + current_candidates[:max_candidates-1]
                else:
                    current_candidates.remove(pos_item_id)
                    current_candidates = [pos_item_id] + current_candidates
                
                for item_id in current_candidates:
                    input_text = construct_reranker_prompt(user_id, item_id, profiles)
                    
                    label = 1 if item_id == pos_item_id else 0
                    
                    batch_examples.append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "input_text": input_text,
                        "label": label,
                    })
        
        examples.extend(batch_examples)
        batch_examples.clear()
        gc.collect()
        logger.info(f"已处理 {min(i + batch_size, total_users)}/{total_users} 用户，当前样本数: {len(examples)}")
    
    logger.info(f"创建了{len(examples)}个训练样本")
    return examples

def tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": examples["label"]
    }

def create_training_examples(profiles, test_data, cf_candidates, num_examples=None, max_pos_items=3, max_candidates=20):
    common_users = set(test_data.keys()) & set(cf_candidates.keys())
    logger.info(f"在测试数据和协同过滤候选项中找到{len(common_users)}个共同用户")
    
    user_ids = list(common_users)
    random.shuffle(user_ids)
    
    if num_examples is not None:
        max_users = min(num_examples, len(user_ids))
        user_ids = user_ids[:max_users]
    
    return create_training_examples_batch(user_ids, profiles, test_data, cf_candidates, 
                                          max_pos_items=max_pos_items, 
                                          max_candidates=max_candidates, 
                                          batch_size=50)

def train_model(args):
    logger.info("开始加载数据...")
    user_profiles = load_profiles(args.profile_path)
    cf_candidates = load_cf_candidates(args.cf_candidates_path)
    test_data = load_test_data(args.test_file_path)
    
    logger.info("创建训练示例...")
    examples = create_training_examples(user_profiles, test_data, cf_candidates, 
                                      args.num_examples,
                                      max_pos_items=args.max_pos_items,
                                      max_candidates=args.max_candidates)
    
    del user_profiles
    del cf_candidates
    del test_data
    gc.collect()
    
    logger.info(f"Loading Qwen2.5-7B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("Loading model for full parameter fine-tuning...")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    local_rank = args.local_rank if args.local_rank != -1 else 0
    
    device_map = {'': local_rank}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        device_map=device_map,
        use_cache=False,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of {total_params} total parameters)")
    
    logger.info("准备数据集...")
    
    chunk_size = 2000
    tokenized_examples = []
    
    for i in range(0, len(examples), chunk_size):
        chunk = examples[i:i+chunk_size]
        chunk_dataset = Dataset.from_dict({
            "user_id": [ex["user_id"] for ex in chunk],
            "item_id": [ex["item_id"] for ex in chunk],
            "input_text": [ex["input_text"] for ex in chunk],
            "label": [ex["label"] for ex in chunk]
        })
        
        tokenized_chunk = chunk_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            batch_size=64,
            remove_columns=chunk_dataset.column_names
        )
        
        tokenized_examples.extend(tokenized_chunk)
        
        del chunk
        del chunk_dataset
        del tokenized_chunk
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"已处理 {min(i + chunk_size, len(examples))}/{len(examples)} 个训练示例")
    
    tokenized_dataset = Dataset.from_list(tokenized_examples)
    
    del examples
    del tokenized_examples
    torch.cuda.empty_cache()
    gc.collect()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    logger.info(f"可用GPU数量: {n_gpus}")
    
    for i in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        logger.info(f"GPU {i} ({gpu_name}) - 已分配内存: {mem_allocated:.2f} GB, 已保留内存: {mem_reserved:.2f} GB")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.03,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
        deepspeed=None,
        local_rank=args.local_rank,
        gradient_checkpointing=True,
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        group_by_length=True,
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
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