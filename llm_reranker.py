import os
import json
import numpy as np
import logging
import torch
from time import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(log_dir, "llm_reranker.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMReranker")

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

class LLMReranker:
    def __init__(self, profile_path=None, local_model_path=None):
        self.local_model_path = "submission/sub/reranker/model"
        self.local_model = None
        self.local_tokenizer = None
        
        self.profiles = {}
        if profile_path:
            self.load_profiles(profile_path)
            
        if local_model_path:
            self._load_local_model()
    
    def _load_local_model(self):
        try:
            logger.info(f"正在从{self.local_model_path}加载本地Qwen模型...")
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            self.local_model = AutoModelForSequenceClassification.from_pretrained(
                self.local_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            
            self.local_model.eval()
            
            logger.info(f"成功加载本地模型: {self.local_model_path}")
            return True
        except Exception as e:
            logger.error(f"加载本地模型失败: {str(e)}")
            return False
    
    def load_profiles(self, profile_path):
        try:
            logger.info(f"正在从{profile_path}加载个人资料...")
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry in data:
                user_id = entry.get('user_id')
                if not user_id:
                    continue
                
                self.profiles[user_id] = {
                    'user_profile': entry.get('user_profile', {}),
                    'positive_items': {},
                    'negative_items': {}
                }
                
                for item in entry.get('positive_items', []):
                    item_id = item.get('item_id')
                    if item_id:
                        self.profiles[user_id]['positive_items'][item_id] = item.get('item_profile', {})
                
                for item in entry.get('negative_items', []):
                    item_id = item.get('item_id')
                    if item_id:
                        self.profiles[user_id]['negative_items'][item_id] = item.get('item_profile', {})
            
            logger.info(f"已加载{len(self.profiles)}个用户的个人资料")
        except Exception as e:
            logger.error(f"加载个人资料时出错: {str(e)}")
            self.profiles = {}
    
    def _construct_prompt(self, user_id, item_id):
        prompt = "You are a Precision Reranking Engine.\n"
        prompt += "Your sole task is to reorder the provided list of candidates strictly and exclusively based on the user preferences detailed below.\n\n"
        
        prompt += "USER INFORMATION\n"
        if user_id in self.profiles:
            profile_data = self.profiles[user_id].get('user_profile', {})
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
        if user_id in self.profiles:
            positive_items = self.profiles[user_id].get('positive_items', {})
            if positive_items:
                prompt += "Items the user previously liked (strong relevance signals):\n"
                for i, (pos_id, profile) in enumerate(list(positive_items.items())[:5]):
                    if pos_id != item_id:
                        summarization = profile.get('summarization', '')
                        if summarization:
                            prompt += f"Item {pos_id}: {summarization}\n"
                prompt += "\n"
            
            negative_items = self.profiles[user_id].get('negative_items', {})
            if negative_items:
                prompt += "Items the user previously disliked (negative signals):\n"
                for i, (neg_id, profile) in enumerate(list(negative_items.items())[:3]):
                    if neg_id != item_id:
                        summarization = profile.get('summarization', '')
                        if summarization:
                            prompt += f"Item {neg_id}: {summarization}\n"
                prompt += "\n"
        
        prompt += "CANDIDATE ITEM DETAILS\n"
        
        if user_id in self.profiles:
            positive_items = self.profiles[user_id].get('positive_items', {})
            negative_items = self.profiles[user_id].get('negative_items', {})
            
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
        
        
        return prompt
            
    def _score_items(self, user_id, item_ids):
        if not self.local_model or not self.local_tokenizer:
            logger.error("本地模型未加载")
            return {}
            
        try:
            scores = {}
            
            with torch.no_grad():
                for item_id in item_ids:
                    prompt = self._construct_prompt(user_id, item_id)
                    
                    inputs = self.local_tokenizer(
                        prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    device = next(self.local_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = self.local_model(**inputs)
                    
                    if outputs.logits.shape[1] > 1:
                        positive_score = outputs.logits[0, 1].item()
                    else:
                        positive_score = outputs.logits[0, 0].item()
                    
                    scores[item_id] = positive_score
                    
            return scores
        except Exception as e:
            logger.error(f"使用本地模型评分时出错: {str(e)}")
            return {}
    
    def rerank(self, candidates, user_id, data_interface):
        start_time = time()
        
        is_numpy = isinstance(candidates, np.ndarray)
        if is_numpy:
            top_n = min(20, len(candidates))
            top_indices = np.argpartition(candidates, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(-candidates[top_indices])]
            
            item_id_map = {idx: data_interface.id2item[idx] for idx in top_indices}
            item_ids = [data_interface.id2item[idx] for idx in top_indices]
            
            original_ordered_ids = [data_interface.id2item[idx] for idx in top_indices]
            original_scores = {data_interface.id2item[idx]: candidates[idx] for idx in top_indices}
            logger.info(f"用户 {user_id} - 原始候选顺序: {original_ordered_ids}")
            logger.info(f"用户 {user_id} - 原始分数: {original_scores}")
            
            logger.info(f"使用本地模型进行重排序...")
            scores = self._score_items(user_id, item_ids)
            
            if not scores:
                logger.warning("模型评分失败，使用原始排序")
                return candidates
            
            try:
                reranked_candidates = candidates.copy()
                
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                reranked_item_ids = [item_id for item_id, _ in sorted_items]
                logger.info(f"用户 {user_id} - 重排序顺序: {reranked_item_ids}")
                
                reverse_ranks = np.array([len(reranked_item_ids) - rank for rank, _ in enumerate(reranked_item_ids)])
                
                softmax_temp = 0.5
                model_probs = softmax(reverse_ranks, temperature=softmax_temp)
                
                valid_scores = [float(score) for _, score in original_scores.items() if score != float('-inf')]
                if valid_scores:
                    highest_score = max(valid_scores)
                    lowest_score = min(valid_scores)
                    score_range = highest_score - lowest_score if highest_score != lowest_score else 1.0
                
                    model_scores = {}
                    for rank, (item_id, prob) in enumerate(zip(reranked_item_ids, model_probs)):
                        model_scores[item_id] = lowest_score + (prob * score_range)
                
                    for rank, item_id in enumerate(reranked_item_ids):
                        if item_id in data_interface.item:
                            idx = data_interface.item[item_id]
                            if idx < len(reranked_candidates):
                                original_score = original_scores.get(item_id, 0)
                                
                                model_score = model_scores[item_id]
                                
                                alpha = 0.7
                                reranked_candidates[idx] = alpha * original_score + (1 - alpha) * model_score
                
                reranked_scores = {data_interface.id2item[idx]: float(reranked_candidates[idx]) 
                                 for idx in top_indices if idx < len(reranked_candidates)}
                
                sorted_reranked_scores = dict(sorted(reranked_scores.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True))
                logger.info(f"用户 {user_id} - 重排序分数（已排序）: {sorted_reranked_scores}")
                
                logger.info(f"重排序完成，耗时 {time() - start_time:.2f} 秒")
                return reranked_candidates
            except Exception as e:
                logger.error(f"应用重排序时出错: {str(e)}")
                return candidates
        else:
            top_n = min(20, len(candidates))
            top_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_indices = [idx for idx, _ in top_items]
            
            item_id_map = {idx: data_interface.id2item[idx] for idx in top_indices}
            item_ids = [data_interface.id2item[idx] for idx in top_indices]
            
            original_ordered_ids = [data_interface.id2item[idx] for idx in top_indices]
            original_scores = {data_interface.id2item[idx]: candidates[idx] for idx in top_indices}
            logger.info(f"用户 {user_id} - 原始候选顺序: {original_ordered_ids}")
            logger.info(f"用户 {user_id} - 原始分数: {original_scores}")
            
            logger.info(f"使用本地模型进行重排序...")
            scores = self._score_items(user_id, item_ids)
            
            if not scores:
                logger.warning("模型评分失败，使用原始排序")
                return candidates
            
            try:
                reranked_candidates = candidates.copy()
                
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                reranked_item_ids = [item_id for item_id, _ in sorted_items]
                logger.info(f"用户 {user_id} - 重排序顺序: {reranked_item_ids}")
                
                reverse_ranks = np.array([len(reranked_item_ids) - rank for rank, _ in enumerate(reranked_item_ids)])
                
                softmax_temp = 0.5
                model_probs = softmax(reverse_ranks, temperature=softmax_temp)
                
                valid_scores = [float(score) for _, score in original_scores.items() if score != float('-inf')]
                if valid_scores:
                    highest_score = max(valid_scores)
                    lowest_score = min(valid_scores)
                    score_range = highest_score - lowest_score if highest_score != lowest_score else 1.0
                
                    model_scores = {}
                    for rank, (item_id, prob) in enumerate(zip(reranked_item_ids, model_probs)):
                        model_scores[item_id] = lowest_score + (prob * score_range)
                
                    for rank, item_id in enumerate(reranked_item_ids):
                        if item_id in data_interface.item:
                            idx = data_interface.item[item_id]
                            if idx in reranked_candidates:
                                original_score = original_scores.get(item_id, 0)
                                
                                model_score = model_scores[item_id]
                                
                                alpha = 0.7
                                reranked_candidates[idx] = alpha * original_score + (1 - alpha) * model_score
                
                reranked_scores = {data_interface.id2item[idx]: float(reranked_candidates[idx]) 
                                 for idx in top_indices if idx in reranked_candidates}
                
                sorted_reranked_scores = dict(sorted(reranked_scores.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True))
                logger.info(f"用户 {user_id} - 重排序分数（已排序）: {sorted_reranked_scores}")
                
                logger.info(f"重排序完成，耗时 {time() - start_time:.2f} 秒")
                return reranked_candidates
            except Exception as e:
                logger.error(f"应用重排序时出错: {str(e)}")
                return candidates

def rerank(candidates, user_id, data_interface, profile_path=None, local_model_path=None):
    reranker = LLMReranker(profile_path=profile_path, local_model_path=local_model_path)
    return reranker.rerank(candidates, user_id, data_interface)