import os
import json
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, tokenizer

def construct_reranker_prompt(user_id, item_id, user_profile=None, item_profile=None, 
                             positive_items=None, negative_items=None):
    prompt = "You are a Precision Reranking Engine.\n"
    prompt += "Your sole task is to reorder the provided list of candidates strictly and exclusively based on the user preferences detailed below.\n\n"
    
    prompt += "USER INFORMATION\n"
    user_summary = ""
    if user_profile and 'summarization' in user_profile:
        user_summary = user_profile['summarization']
    
    if user_summary:
        prompt += f"User Information: User {user_id} with explicit preferences: {user_summary}\n\n"
    else:
        prompt += f"User Information: User {user_id}\n\n"
    
    prompt += "CANDIDATE ITEMS\n"
    prompt += f"Candidate Items List: {item_id}\n"
    prompt += f"Original Ranking: {item_id}\n\n"
    
    prompt += "USER INTERACTION HISTORY\n"
    if positive_items:
        prompt += "Items the user previously liked (strong relevance signals):\n"
        count = 0
        for pos_id, pos_profile in positive_items.items():
            if pos_id != item_id and count < 5:
                summarization = pos_profile.get('summarization', '')
                if summarization:
                    prompt += f"Item {pos_id}: {summarization}\n"
                    count += 1
        prompt += "\n"
    
    if negative_items:
        prompt += "Items the user previously disliked (negative signals):\n"
        count = 0
        for neg_id, neg_profile in negative_items.items():
            if neg_id != item_id and count < 3:
                summarization = neg_profile.get('summarization', '')
                if summarization:
                    prompt += f"Item {neg_id}: {summarization}\n"
                    count += 1
        prompt += "\n"
    
    prompt += "CANDIDATE ITEM DETAILS\n"
    
    item_signal = ""
    item_summary = ""
    
    if item_profile:
        item_summary = item_profile.get('summarization', f"Item {item_id}")
    else:
        item_summary = f"Item {item_id}"
    
    prompt += f"Item {item_id} {item_signal}: {item_summary}\n\n"
    
    prompt += "RANKING REASONING PROCESS\n"
    prompt += "Follow this reasoning process:\n"
    prompt += "Step 1: Prioritize items that closely match the user's historical preferences and explicit likes.\n"
    prompt += "Step 2: Demote items similar to those the user has historically disliked.\n"
    prompt += "Step 3: Make minimal adjustments to the original ranking, changing positions only when you have high confidence.\n\n"
    
    prompt += "RANKING GUIDELINES\n"
    prompt += "1. Promote the most relevant items to top positions\n"
    prompt += "2. Make conservative adjustments - only move items when necessary\n"
    prompt += "3. The original ranking is usually correct - only adjust with clear evidence\n\n"
    
    return prompt

def score_items(model, tokenizer, user_id, items_to_score, profiles_data, max_length=512, batch_size=4):
    device = next(model.parameters()).device
    scores = {}
    
    user_profile = None
    positive_items = None
    negative_items = None
    
    user_data = profiles_data.get(user_id, {})
    if user_data:
        user_profile = user_data.get('user_profile', {})
        positive_items = user_data.get('positive_items', {})
        negative_items = user_data.get('negative_items', {})
    
    prompts = []
    item_ids = []
    
    for item_id in items_to_score:
        item_profile = None
        if item_id in positive_items:
            item_profile = positive_items[item_id]
        elif item_id in negative_items:
            item_profile = negative_items[item_id]
        
        prompt = construct_reranker_prompt(
            user_id, 
            item_id, 
            user_profile, 
            item_profile, 
            positive_items, 
            negative_items
        )
        
        prompts.append(prompt)
        item_ids.append(item_id)
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_item_ids = item_ids[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            if outputs.logits.shape[1] > 1:
                batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            else:
                batch_scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()
        
        for j, item_id in enumerate(batch_item_ids):
            scores[item_id] = float(batch_scores[j])
    
    return scores

def load_merged_profiles(profile_path):
    with open(profile_path, 'r', encoding='utf-8') as f:
        profiles_data = json.load(f)
    
    profiles_dict = {}
    for profile in profiles_data:
        user_id = profile.get('user_id')
        if not user_id:
            continue
        
        profiles_dict[user_id] = {
            'user_profile': profile.get('user_profile', {}),
            'positive_items': {},
            'negative_items': {}
        }
        
        for pos_item in profile.get('positive_items', []):
            item_id = pos_item.get('item_id')
            if item_id:
                profiles_dict[user_id]['positive_items'][item_id] = pos_item.get('item_profile', {})
        
        for neg_item in profile.get('negative_items', []):
            item_id = neg_item.get('item_id')
            if item_id:
                profiles_dict[user_id]['negative_items'][item_id] = neg_item.get('item_profile', {})
    
    return profiles_dict

def load_cf_candidates(cf_path):
    with open(cf_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def rerank_candidates(model, tokenizer, user_id, candidates, profiles_data, max_items=20):
    candidates_to_score = candidates[:max_items] if len(candidates) > max_items else candidates
    
    scores = score_items(
        model, 
        tokenizer, 
        user_id, 
        candidates_to_score, 
        profiles_data
    )
    
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    reranked_candidates = [item_id for item_id, _ in sorted_items]
    
    return reranked_candidates

def main(args):
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    
    logger.info(f"Loading profiles from {args.profile_path}")
    profiles_data = load_merged_profiles(args.profile_path)
    
    logger.info(f"Loading candidates from {args.candidates_path}")
    cf_candidates = load_cf_candidates(args.candidates_path)
    
    results = {}
    
    for user_id, candidates in tqdm(cf_candidates.items(), desc="Reranking"):
        reranked = rerank_candidates(
            model, 
            tokenizer, 
            user_id, 
            candidates, 
            profiles_data,
            max_items=args.max_items
        )
        results[user_id] = reranked
    
    output_file = args.output_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Reranked results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank candidates using Qwen2.5 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--profile_path", type=str, required=True, help="Path to merged profiles JSON file")
    parser.add_argument("--candidates_path", type=str, required=True, help="Path to CF candidates JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for reranked results")
    parser.add_argument("--max_items", type=int, default=20, help="Maximum items to rerank per user")
    
    args = parser.parse_args()
    main(args) 