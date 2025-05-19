import torch
import torch.nn.functional as F
import torch.nn as nn


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = ((user_emb-pos_item_emb)**2).sum(dim=1)
    neg_score = ((user_emb-neg_item_emb)**2).sum(dim=1)
    loss = F.relu(pos_score-neg_score+0.5)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
    def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    if sim == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)

    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return F.cross_entropy(logits, labels)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

def kl_div_loss(p, q):
    p = F.normalize(p, p=2, dim=1)
    q = F.normalize(q, p=2, dim=1)
    
    kl_div = -F.kl_div(
        F.log_softmax(p, dim=1),
        F.softmax(q, dim=1),
        reduction='batchmean'
    )
    return kl_div

def pos_kl_div_loss(p, q):
    p = F.normalize(p, p=2, dim=1)
    q = F.normalize(q, p=2, dim=1)
    
    kl_div = -F.kl_div(
        F.log_softmax(p, dim=1),
        F.softmax(q, dim=1),
        reduction='batchmean'
    )
    return kl_div

def js_div_loss(p, q):
    p = F.normalize(p, p=2, dim=1)
    q = F.normalize(q, p=2, dim=1)
    
    m = (F.softmax(p, dim=1) + F.softmax(q, dim=1)) / 2
    
    kl_p_m = F.kl_div(
        F.log_softmax(p, dim=1),
        m,
        reduction='batchmean'
    )
    
    kl_q_m = F.kl_div(
        F.log_softmax(q, dim=1),
        m,
        reduction='batchmean'
    )
    
    return -(kl_p_m + kl_q_m) / 2

def pos_js_div_loss(p, q):
    p = F.normalize(p, p=2, dim=1)
    q = F.normalize(q, p=2, dim=1)
    
    m = (F.softmax(p, dim=1) + F.softmax(q, dim=1)) / 2
    
    kl_p_m = F.kl_div(
        F.log_softmax(p, dim=1),
        m,
        reduction='batchmean'
    )
    
    kl_q_m = F.kl_div(
        F.log_softmax(q, dim=1),
        m,
        reduction='batchmean'
    )
    
    return (kl_p_m + kl_q_m) / 2

def wasserstein_distance(scores1, scores2):
    sorted1, _ = torch.sort(scores1)
    sorted2, _ = torch.sort(scores2)
    return torch.mean(torch.abs(sorted1 - sorted2))