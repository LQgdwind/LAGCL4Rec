import torch
import torch.nn as nn
import torch.nn.functional as F
from recommender import GraphRecommender
from conf import OptionConf
from sampler import next_batch_pairwise_with_groups
from graph import TorchGraphInterface
from loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import numpy as np
from tqdm import tqdm


class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['XSimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        
        self.hard_weight = 2
        self.user_cl_weight = 1.0
        self.item_cl_weight = 1.0
        
        self.temp_pos = self.temp
        self.temp_easy = self.temp * 0.25
        self.temp_hard = self.temp * 0.1
        
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl)
        
        try:
            self.user_groups = self.load_group_info('/home/zlq/submission/EMNLP2025/dataset/ml-1m/user_group.txt')
            self.item_groups = self.load_group_info('/home/zlq/submission/EMNLP2025/dataset/ml-1m/item_group.txt')
        except Exception as e:
            print(f"Error loading group files: {e}")
            print("Using default grouping (all entities in one group)")
            self.user_groups = {uid: 0 for uid in range(self.data.user_num)}
            self.item_groups = {iid: 0 for iid in range(self.data.item_num)}
    
    def load_group_info(self, filepath):
        group_dict = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        entity_id, group_id = int(parts[0]), int(parts[1])
                        group_dict[entity_id] = group_id
        except Exception as e:
            print(f"Error loading group file {filepath}: {e}")
            return None
            
        return group_dict
    
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        batch_count = self.data.user_num // self.batch_size + 1
        
        for epoch in range(self.maxEpoch):
            try:
                training_batches = tqdm(next_batch_pairwise_with_groups(
                    self.data, self.batch_size, self.user_groups, self.item_groups, 
                    n_easy_negs=1, n_hard_negs=1), total=batch_count, 
                    desc=f"Epoch {epoch+1}/{self.maxEpoch}")
            except Exception as e:
                print(f"Error creating training batches: {e}")
                print("Using default groups for all users/items")
                user_groups_default = [0] * len(self.data.user)
                item_groups_default = [0] * len(self.data.item)
                training_batches = tqdm(next_batch_pairwise_with_groups(
                    self.data, self.batch_size, user_groups_default, item_groups_default, 
                    n_easy_negs=1, n_hard_negs=1), total=batch_count, 
                    desc=f"Epoch {epoch+1}/{self.maxEpoch}")
                
            for n, batch in enumerate(training_batches):
                try:
                    user_idx, pos_idx, easy_neg_idx, hard_neg_idx, user_groups_batch, item_groups_batch = batch
                    
                    rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)
                    
                    user_emb = rec_user_emb[user_idx]
                    pos_item_emb = rec_item_emb[pos_idx]
                    easy_neg_emb = rec_item_emb[easy_neg_idx]
                    hard_neg_emb = rec_item_emb[hard_neg_idx]
                    
                    easy_rec_loss = bpr_loss(user_emb, pos_item_emb, easy_neg_emb)
                    hard_rec_loss = bpr_loss(user_emb, pos_item_emb, hard_neg_emb)
                    rec_loss = easy_rec_loss + self.hard_weight * hard_rec_loss
                    
                    cl_loss = self.cl_rate * self.cal_group_aware_cl_loss(
                        user_idx, pos_idx, user_groups_batch, item_groups_batch,
                        rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb
                    )
                    
                    batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                    
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    
                    if n % 50 == 0 and n > 0:
                        training_batches.set_postfix(
                            easy_loss=f"{easy_rec_loss.item():.4f}",
                            hard_loss=f"{hard_rec_loss.item():.4f}",
                            cl_loss=f"{cl_loss.item():.4f}"
                        )
                except Exception as e:
                    print(f"Error processing batch {n}: {e}")
                    continue
                    
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                
            self.fast_evaluation(epoch)
            
        if hasattr(self, 'best_user_emb') and hasattr(self, 'best_item_emb'):
            self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    
    def cal_group_aware_cl_loss(self, user_idx, item_idx, user_groups, item_groups, 
                               user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.tensor(list(set(user_idx)), dtype=torch.long).cuda()
        i_idx = torch.tensor(list(set(item_idx)), dtype=torch.long).cuda()
        
        user_idx_to_pos = {idx: pos for pos, idx in enumerate(user_idx)}
        item_idx_to_pos = {idx: pos for pos, idx in enumerate(item_idx)}
        
        batch_user_groups = []
        for idx in u_idx:
            user_id = idx.item()
            if user_id in user_idx_to_pos:
                batch_pos = user_idx_to_pos[user_id]
                if batch_pos < len(user_groups):
                    batch_user_groups.append(user_groups[batch_pos])
                elif user_id in self.user_groups:
                    batch_user_groups.append(self.user_groups[user_id])
                else:
                    batch_user_groups.append(0)
            else:
                batch_user_groups.append(0)
        
        batch_item_groups = []
        for idx in i_idx:
            item_id = idx.item()
            if item_id in item_idx_to_pos:
                batch_pos = item_idx_to_pos[item_id]
                if batch_pos < len(item_groups):
                    batch_item_groups.append(item_groups[batch_pos])
                elif item_id in self.item_groups:
                    batch_item_groups.append(self.item_groups[item_id])
                else:
                    batch_item_groups.append(0)
            else:
                batch_item_groups.append(0)
        
        batch_user_groups = torch.tensor(batch_user_groups, dtype=torch.long).cuda()
        batch_item_groups = torch.tensor(batch_item_groups, dtype=torch.long).cuda()
        
        user_cl_loss = self.group_aware_InfoNCE(
            user_view1[u_idx], user_view2[u_idx], 
            self.temp_pos, self.temp_easy, self.temp_hard, batch_user_groups
        )
        
        item_cl_loss = self.group_aware_InfoNCE(
            item_view1[i_idx], item_view2[i_idx], 
            self.temp_pos, self.temp_easy, self.temp_hard, batch_item_groups
        )
        
        return self.user_cl_weight * user_cl_loss + self.item_cl_weight * item_cl_loss
    
    def group_aware_InfoNCE(self, view1, view2, temp_pos, temp_easy, temp_hard, group_ids):
        batch_size = view1.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0).cuda()
        
        view1_norm = F.normalize(view1, dim=1)
        view2_norm = F.normalize(view2, dim=1)
        
        sim_matrix = torch.matmul(view1_norm, view2_norm.T)
        
        group_ids_matrix = group_ids.view(-1, 1)
        same_group_mask = (group_ids_matrix == group_ids.view(1, -1)).float()
        
        diag_mask = torch.eye(batch_size).cuda()
        same_group_mask = same_group_mask * (1 - diag_mask)
        
        all_same_group = (same_group_mask.sum() == batch_size * (batch_size - 1))
        
        if all_same_group:
            effective_temp_pos = max(temp_pos, 0.1)
            logits = sim_matrix / effective_temp_pos
            
            labels = torch.arange(batch_size).cuda()
            loss = F.cross_entropy(logits, labels)
            return loss
        
        diff_group_mask = 1 - same_group_mask - diag_mask
        
        if diff_group_mask.sum() == 0:
            diff_group_mask = torch.rand(batch_size, batch_size).cuda() * 0.01
            diff_group_mask = diff_group_mask * (1 - same_group_mask - diag_mask)
        
        pos_mask = diag_mask
        
        effective_temp_pos = max(temp_pos, 0.1)
        effective_temp_easy = max(temp_easy, 0.1)
        effective_temp_hard = max(temp_hard, 0.1)
        
        pos_sim = sim_matrix * pos_mask / effective_temp_pos
        
        hard_neg_sim = sim_matrix * same_group_mask / effective_temp_hard
        
        easy_neg_sim = sim_matrix * diff_group_mask / effective_temp_easy
        
        combined_logits = torch.zeros_like(sim_matrix)
        combined_logits = combined_logits + pos_sim + hard_neg_sim + easy_neg_sim
        
        max_logits, _ = torch.max(combined_logits, dim=1, keepdim=True)
        logits_stable = combined_logits - max_logits
        
        exp_logits = torch.exp(logits_stable)
        
        pos_exp_logits = torch.sum(exp_logits * pos_mask, dim=1)
        
        weighted_hard_exp_logits = exp_logits * same_group_mask * self.hard_weight
        easy_exp_logits = exp_logits * diff_group_mask
        
        denominator = pos_exp_logits + torch.sum(weighted_hard_exp_logits + easy_exp_logits, dim=1) + 1e-12
        
        contrastive_loss = -torch.log(pos_exp_logits / denominator + 1e-12)
        
        if torch.isnan(contrastive_loss).any() or torch.isinf(contrastive_loss).any():
            print(f"警告: 损失函数出现NaN或Inf值")
            print(f"温度参数: pos={effective_temp_pos}, easy={effective_temp_easy}, hard={effective_temp_hard}")
            print(f"同组样本数: {same_group_mask.sum().item()}, 异组样本数: {diff_group_mask.sum().item()}")
            print(f"相似度矩阵范围: [{sim_matrix.min().item()}, {sim_matrix.max().item()}]")
            return torch.tensor(0.1).cuda()
        
        return torch.mean(contrastive_loss)

    def predict(self, user):
        try:
            if not hasattr(self, 'user_emb') or not hasattr(self, 'item_emb'):
                with torch.no_grad():
                    self.user_emb, self.item_emb = self.model()
            
            u_idx = self.data.get_user_id(user)
            
            with torch.no_grad():
                user_tensor = self.user_emb[u_idx].unsqueeze(0)
                scores = torch.matmul(user_tensor, self.item_emb.transpose(0, 1)).squeeze().cpu().numpy()
            
            return scores
            
        except Exception as e:
            print(f"Error predicting for user {user}: {e}")
            return np.zeros(self.data.item_num)
            
    def save(self):
        if hasattr(self, 'user_emb') and hasattr(self, 'item_emb'):
            self.best_user_emb = self.user_emb
            self.best_item_emb = self.item_emb


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                noise_norm = F.normalize(random_noise, dim=-1) 
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * noise_norm * self.eps
            all_embeddings.append(ego_embeddings)
            if k == self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
