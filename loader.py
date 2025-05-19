import os.path
from os import remove
from re import split
import os
from random import shuffle

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_train_set(file, rec_type):
        data = []
        pos_data = []
        neg_data = []
        with open(file) as f:
            for line in f:
                items = split('\t', line.strip())
                user_id = items[0]
                item_id = items[1]
                weight = items[2]
                data.append([user_id, item_id, float(weight)])
                if float(weight) > 0:
                    pos_data.append([user_id, item_id, float(weight)])
                else:
                    neg_data.append([user_id, item_id, float(weight)])
        print(f'pos_data: {len(pos_data)}')
        print(f'neg_data: {len(neg_data)}')
        print(f'data: {len(data)}')
        return pos_data, neg_data, data

    @staticmethod
    def load_test_set(file, rec_type):
        data = []
        with open(file) as f:
            for line in f:
                items = split('\t', line.strip())
                user_id = items[0]
                item_id = items[1]
                weight = items[2]
                if float(weight) > 0:
                    data.append([user_id, item_id, float(weight)])

        return data

    
    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

def process_data(dir_path,file_name,split_ratio,tag_count=None):
    train_data = []
    test_data = []
    all_data = []
    
    with open(dir_path+'/'+file_name) as f:
        for ind,line in enumerate(f):
            if '"' in line:
                continue
            if len(line.strip().split(',')) < 3:
                continue
            items = line.strip().split(',')
            data_slice = [items[0],items[1],float(items[2])]
            all_data.append(data_slice)
    
    shuffle(all_data)
    if split_ratio == -1:
        result = [[], all_data]
        return result
    
    train_size = int(len(all_data) * split_ratio)
    train_data = all_data[0:train_size]
    test_data = all_data[train_size:]
    
    return [train_data,test_data]

def load_data(config,benchmark=False,from_disk=False):
    train_data = []
    test_data = []
    
    if from_disk:
        train_path = config['training.set']
        test_path = config['test.set']
        train_name = train_path.split('/')[-1]
        test_name = test_path.split('/')[-1]
        train_path = '/'.join(train_path.split('/')[:-1])
        test_path = '/'.join(test_path.split('/')[:-1])
        
        if train_name != test_name:
            train_data = process_data(train_path,train_name,-1)[1]
            test_data = process_data(test_path,test_name,-1)[1]
        else:
            data = process_data(train_path,train_name,float(config['training.set.ratio']))
            train_data = data[0]
            test_data = data[1]
    
    else:
        file_dir = config['data.input.path']
        file_name = config['data.input.dataset']
        
        data = process_data(file_dir,file_name,float(config['training.set.ratio']))
        train_data = data[0]
        test_data = data[1]
    
    return train_data, test_data
