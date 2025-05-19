from random import shuffle,randint,choice,sample
import numpy as np

def next_batch_pairwise_with_groups(data, batch_size, user_groups, item_groups, n_easy_negs=1, n_hard_negs=1):

    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    
    group_to_items = {}
    for item_id, group_id in item_groups.items():
        if group_id not in group_to_items:
            group_to_items[group_id] = []
        if item_id in data.item:
            group_to_items[group_id].append(item_id)
    
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
            
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        
        u_idx, i_idx, easy_j_idx, hard_j_idx = [], [], [], []
        user_groups_batch, item_groups_batch = [], []
        item_list = list(data.item.keys())
        
        for i, user in enumerate(users):
            user_idx = data.user[user]
            item_idx = data.item[items[i]]
            
            u_idx.append(user_idx)
            i_idx.append(item_idx)
            
            user_group = user_groups.get(user, 0)
            item_group = item_groups.get(items[i], 0)
            
            user_groups_batch.append(user_group)
            item_groups_batch.append(item_group)
            
            pos_item = items[i]
            pos_item_group = item_groups.get(pos_item, -1)
            
            for _ in range(n_easy_negs):
                neg_item = choice(item_list)
                while (neg_item in data.training_set_u[user] or 
                      item_groups.get(neg_item, -2) == pos_item_group):
                    neg_item = choice(item_list)
                easy_j_idx.append(data.item[neg_item])
            
            if pos_item_group in group_to_items and len(group_to_items[pos_item_group]) > 1:
                same_group_items = group_to_items[pos_item_group]
                for _ in range(n_hard_negs):
                    neg_item = choice(same_group_items)
                    attempts = 0
                    while neg_item in data.training_set_u[user] and attempts < 10:
                        neg_item = choice(same_group_items)
                        attempts += 1
                    
                    if neg_item in data.training_set_u[user]:
                        neg_item = choice(item_list)
                        while neg_item in data.training_set_u[user]:
                            neg_item = choice(item_list)
                    
                    hard_j_idx.append(data.item[neg_item])
            else:
                for _ in range(n_hard_negs):
                    neg_item = choice(item_list)
                    while neg_item in data.training_set_u[user]:
                        neg_item = choice(item_list)
                    hard_j_idx.append(data.item[neg_item])
        
        yield u_idx, i_idx, easy_j_idx, hard_j_idx, user_groups_batch, item_groups_batch
