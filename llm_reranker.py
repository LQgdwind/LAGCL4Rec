import os
import json
import numpy as np
import logging
import torch
from time import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 设置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(log_dir, "llm_reranker.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMReranker")

# 添加softmax函数
def softmax(x, temperature=1.0):
    """
    Compute softmax values for each set of scores in x.
    温度参数控制分布的平滑度，较高的温度使分布更平滑
    """
    e_x = np.exp((x - np.max(x)) / temperature)  # 减去最大值以防止数值溢出并应用温度
    return e_x / e_x.sum()

class LLMReranker:
    def __init__(self, profile_path=None, local_model_path=None):
        """
        Initialize Qwen reranker with local model and profile data
        
        Args:
            profile_path: Path to the merged profiles JSON file
            local_model_path: Path to local Qwen model
        """
        # 本地模型设置
        self.local_model_path = "submission/sub/reranker/model"
        self.local_model = None
        self.local_tokenizer = None
        
        # 加载配置文件
        self.profiles = {}
        if profile_path:
            self.load_profiles(profile_path)
            
        # 加载本地模型
        if local_model_path:
            self._load_local_model()
    
    def _load_local_model(self):
        """加载微调后的本地Qwen模型用于重排序"""
        try:
            logger.info(f"正在从{self.local_model_path}加载本地Qwen模型...")
            
            # 加载分类模型和tokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            self.local_model = AutoModelForSequenceClassification.from_pretrained(
                self.local_model_path,
                device_map="auto",  # 自动选择可用设备
                torch_dtype=torch.bfloat16  # 使用bfloat16减少内存占用
            )
            
            # 设置为评估模式
            self.local_model.eval()
            
            logger.info(f"成功加载本地模型: {self.local_model_path}")
            return True
        except Exception as e:
            logger.error(f"加载本地模型失败: {str(e)}")
            return False
    
    def load_profiles(self, profile_path):
        """加载用户和物品的个人资料"""
        try:
            logger.info(f"正在从{profile_path}加载个人资料...")
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建查找字典以便快速访问
            for entry in data:
                user_id = entry.get('user_id')
                if not user_id:
                    continue
                
                # 存储用户资料
                self.profiles[user_id] = {
                    'user_profile': entry.get('user_profile', {}),
                    'positive_items': {},
                    'negative_items': {}
                }
                
                # 存储正面项目
                for item in entry.get('positive_items', []):
                    item_id = item.get('item_id')
                    if item_id:
                        self.profiles[user_id]['positive_items'][item_id] = item.get('item_profile', {})
                
                # 存储负面项目
                for item in entry.get('negative_items', []):
                    item_id = item.get('item_id')
                    if item_id:
                        self.profiles[user_id]['negative_items'][item_id] = item.get('item_profile', {})
            
            logger.info(f"已加载{len(self.profiles)}个用户的个人资料")
        except Exception as e:
            logger.error(f"加载个人资料时出错: {str(e)}")
            self.profiles = {}
    
    def _construct_prompt(self, user_id, item_id):
        """
        构建用于评估单个物品的提示，使用新的格式
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            构建好的提示
        """
        # 创建提示
        prompt = "You are a Precision Reranking Engine.\n"
        prompt += "Your sole task is to reorder the provided list of candidates strictly and exclusively based on the user preferences detailed below.\n\n"
        
        # 用户信息部分
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
        
        # 候选项部分
        prompt += "CANDIDATE ITEMS\n"
        prompt += f"Candidate Items List: {item_id}\n"
        prompt += f"Original Ranking: {item_id}\n\n"
        
        # 用户交互历史部分
        prompt += "USER INTERACTION HISTORY\n"
        if user_id in self.profiles:
            positive_items = self.profiles[user_id].get('positive_items', {})
            if positive_items:
                prompt += "Items the user previously liked (strong relevance signals):\n"
                for i, (pos_id, profile) in enumerate(list(positive_items.items())[:5]):
                    if pos_id != item_id:  # 避免重复当前物品
                        summarization = profile.get('summarization', '')
                        if summarization:
                            prompt += f"Item {pos_id}: {summarization}\n"
                prompt += "\n"
            
            # 历史信息 - 用户不喜欢的物品
            negative_items = self.profiles[user_id].get('negative_items', {})
            if negative_items:
                prompt += "Items the user previously disliked (negative signals):\n"
                for i, (neg_id, profile) in enumerate(list(negative_items.items())[:3]):
                    if neg_id != item_id:  # 避免重复当前物品
                        summarization = profile.get('summarization', '')
                        if summarization:
                            prompt += f"Item {neg_id}: {summarization}\n"
                prompt += "\n"
        
        # 候选物品详情
        prompt += "CANDIDATE ITEM DETAILS\n"
        
        # 获取物品资料
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
        
        # 排序推理过程
        prompt += "RANKING REASONING PROCESS\n"
        prompt += "Follow this reasoning process:\n"
        prompt += "Step 1: Prioritize items that closely match the user's historical preferences and explicit likes.\n"
        prompt += "Step 2: Demote items similar to those the user has historically disliked.\n"
        prompt += "Step 3: Make minimal adjustments to the original ranking, changing positions only when you have high confidence.\n\n"
        
        # 排序指南
        prompt += "RANKING GUIDELINES\n"
        prompt += "1. Promote the most relevant items to top positions\n"
        prompt += "2. Make conservative adjustments - only move items when necessary\n"
        prompt += "3. The original ranking is usually correct - only adjust with clear evidence\n\n"
        
        
        return prompt
            
    def _score_items(self, user_id, item_ids):
        """
        使用本地模型对用户-物品对打分
        
        Args:
            user_id: 用户ID
            item_ids: 物品ID列表
            
        Returns:
            字典 {item_id: relevance_score}
        """
        if not self.local_model or not self.local_tokenizer:
            logger.error("本地模型未加载")
            return {}
            
        try:
            scores = {}
            
            # 对每个物品计算得分
            with torch.no_grad():  # 不计算梯度，减少内存使用
                for item_id in item_ids:
                    # 构建提示
                    prompt = self._construct_prompt(user_id, item_id)
                    
                    # 准备输入
                    inputs = self.local_tokenizer(
                        prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=512,  # 根据需要调整最大长度
                        return_tensors="pt"
                    )
                    
                    # 将输入移到模型所在设备
                    device = next(self.local_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 获取模型输出
                    outputs = self.local_model(**inputs)
                    
                    # 获取积极类别的得分
                    if outputs.logits.shape[1] > 1:  # 二分类情况
                        positive_score = outputs.logits[0, 1].item()  # 索引1是积极类别
                    else:  # 单分类情况
                        positive_score = outputs.logits[0, 0].item()
                    
                    # 保存得分
                    scores[item_id] = positive_score
                    
            return scores
        except Exception as e:
            logger.error(f"使用本地模型评分时出错: {str(e)}")
            return {}
    
    def rerank(self, candidates, user_id, data_interface):
        """
        使用本地模型重排序候选项
        
        Args:
            candidates: numpy数组或字典 {item_idx: score}
            user_id: 用户ID
            data_interface: 数据接口对象，包含项目映射方法
            
        Returns:
            重排序后的候选项（与输入类型相同）
        """
        start_time = time()
        
        # 处理不同的输入类型
        is_numpy = isinstance(candidates, np.ndarray)
        if is_numpy:
            # 获取前N个候选项进行重排序（限制为前20个以提高效率）
            top_n = min(20, len(candidates))
            top_indices = np.argpartition(candidates, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(-candidates[top_indices])]
            
            # 创建内部ID到原始项目ID的映射
            item_id_map = {idx: data_interface.id2item[idx] for idx in top_indices}
            item_ids = [data_interface.id2item[idx] for idx in top_indices]
            
            # 记录原始排序顺序和分数
            original_ordered_ids = [data_interface.id2item[idx] for idx in top_indices]
            original_scores = {data_interface.id2item[idx]: candidates[idx] for idx in top_indices}
            logger.info(f"用户 {user_id} - 原始候选顺序: {original_ordered_ids}")
            logger.info(f"用户 {user_id} - 原始分数: {original_scores}")
            
            # 使用本地模型评分
            logger.info(f"使用本地模型进行重排序...")
            scores = self._score_items(user_id, item_ids)
            
            # 如果评分失败，返回原始候选项
            if not scores:
                logger.warning("模型评分失败，使用原始排序")
                return candidates
            
            try:
                # 创建一个新的数组用于重排序后的分数
                reranked_candidates = candidates.copy()
                
                # 根据得分对物品进行排序
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                reranked_item_ids = [item_id for item_id, _ in sorted_items]
                logger.info(f"用户 {user_id} - 重排序顺序: {reranked_item_ids}")
                
                # 使用softmax处理模型的排名得分
                # 1. 创建反向排名得分（排名越前，得分越高）
                reverse_ranks = np.array([len(reranked_item_ids) - rank for rank, _ in enumerate(reranked_item_ids)])
                
                # 2. 应用softmax获得归一化的概率分布 - 使用较低温度使分布更陡峭
                softmax_temp = 0.5  # 降低温度使分布更陡峭，增大前排项目的得分差异
                model_probs = softmax(reverse_ranks, temperature=softmax_temp)
                
                # 3. 将softmax后的分数映射到原始分数的取值范围
                valid_scores = [score for _, score in original_scores.items() if score != float('-inf')]
                if valid_scores:
                    highest_score = max(valid_scores)
                    lowest_score = min(valid_scores)
                    score_range = highest_score - lowest_score if highest_score != lowest_score else 1.0
                
                    # 调整后的模型分数
                    model_scores = {}
                    for rank, (item_id, prob) in enumerate(zip(reranked_item_ids, model_probs)):
                        # 将概率映射到原始分数的范围
                        model_scores[item_id] = lowest_score + (prob * score_range)
                
                    # 对原始分数和模型分数进行加权平均
                    for rank, item_id in enumerate(reranked_item_ids):
                        if item_id in data_interface.item:  # 确保物品存在于映射中
                            idx = data_interface.item[item_id]
                            if idx < len(reranked_candidates):
                                # 获取原始分数
                                original_score = original_scores.get(item_id, 0)
                                
                                # 获取模型分数
                                model_score = model_scores[item_id]
                                
                                # 对原始分数和模型分数进行加权平均
                                alpha = 0.7  # 原始分数的权重
                                reranked_candidates[idx] = alpha * original_score + (1 - alpha) * model_score
                
                # 记录重排序后的得分
                reranked_scores = {data_interface.id2item[idx]: reranked_candidates[idx] 
                                 for idx in top_indices if idx < len(reranked_candidates)}
                
                # 按分数从高到低对重排序后的得分进行排序，方便查看
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
            # 处理字典输入
            top_n = min(20, len(candidates))
            top_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_indices = [idx for idx, _ in top_items]
            
            # 创建内部ID到原始项目ID的映射
            item_id_map = {idx: data_interface.id2item[idx] for idx in top_indices}
            item_ids = [data_interface.id2item[idx] for idx in top_indices]
            
            # 记录原始排序顺序和分数
            original_ordered_ids = [data_interface.id2item[idx] for idx in top_indices]
            original_scores = {data_interface.id2item[idx]: candidates[idx] for idx in top_indices}
            logger.info(f"用户 {user_id} - 原始候选顺序: {original_ordered_ids}")
            logger.info(f"用户 {user_id} - 原始分数: {original_scores}")
            
            # 使用本地模型评分
            logger.info(f"使用本地模型进行重排序...")
            scores = self._score_items(user_id, item_ids)
            
            # 如果评分失败，返回原始候选项
            if not scores:
                logger.warning("模型评分失败，使用原始排序")
                return candidates
            
            try:
                # 创建原始候选项字典的副本
                reranked_candidates = candidates.copy()
                
                # 根据得分对物品进行排序
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                reranked_item_ids = [item_id for item_id, _ in sorted_items]
                logger.info(f"用户 {user_id} - 重排序顺序: {reranked_item_ids}")
                
                # 使用softmax处理模型的排名得分
                # 1. 创建反向排名得分（排名越前，得分越高）
                reverse_ranks = np.array([len(reranked_item_ids) - rank for rank, _ in enumerate(reranked_item_ids)])
                
                # 2. 应用softmax获得归一化的概率分布 - 使用较低温度使分布更陡峭
                softmax_temp = 0.5  # 降低温度使分布更陡峭，增大前排项目的得分差异
                model_probs = softmax(reverse_ranks, temperature=softmax_temp)
                
                # 3. 将softmax后的分数映射到原始分数的取值范围
                valid_scores = [score for _, score in original_scores.items() if score != float('-inf')]
                if valid_scores:
                    highest_score = max(valid_scores)
                    lowest_score = min(valid_scores)
                    score_range = highest_score - lowest_score if highest_score != lowest_score else 1.0
                
                    # 调整后的模型分数
                    model_scores = {}
                    for rank, (item_id, prob) in enumerate(zip(reranked_item_ids, model_probs)):
                        # 将概率映射到原始分数的范围
                        model_scores[item_id] = lowest_score + (prob * score_range)
                
                    # 对原始分数和模型分数进行加权平均
                    for rank, item_id in enumerate(reranked_item_ids):
                        if item_id in data_interface.item:  # 确保物品存在于映射中
                            idx = data_interface.item[item_id]
                            if idx in reranked_candidates:
                                # 获取原始分数
                                original_score = original_scores.get(item_id, 0)
                                
                                # 获取模型分数
                                model_score = model_scores[item_id]
                                
                                # 对原始分数和模型分数进行加权平均
                                alpha = 0.7  # 原始分数的权重
                                reranked_candidates[idx] = alpha * original_score + (1 - alpha) * model_score
                
                # 记录重排序后的得分
                reranked_scores = {data_interface.id2item[idx]: reranked_candidates[idx] 
                                 for idx in top_indices if idx in reranked_candidates}
                
                # 按分数从高到低对重排序后的得分进行排序，方便查看
                sorted_reranked_scores = dict(sorted(reranked_scores.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True))
                logger.info(f"用户 {user_id} - 重排序分数（已排序）: {sorted_reranked_scores}")
                
                logger.info(f"重排序完成，耗时 {time() - start_time:.2f} 秒")
                return reranked_candidates
            except Exception as e:
                logger.error(f"应用重排序时出错: {str(e)}")
                return candidates

# 更新重排序函数
def rerank(candidates, user_id, data_interface, profile_path=None, local_model_path=None):
    """
    使用本地模型重排序候选项
    
    Args:
        candidates: numpy数组或字典 {item_idx: score}
        user_id: 用户ID
        data_interface: 数据接口对象，包含项目映射方法
        profile_path: 合并的个人资料JSON文件的路径
        local_model_path: 本地模型的路径
        
    Returns:
        重排序后的候选项（与输入类型相同）
    """
    reranker = LLMReranker(profile_path=profile_path, local_model_path=local_model_path)
    return reranker.rerank(candidates, user_id, data_interface) 