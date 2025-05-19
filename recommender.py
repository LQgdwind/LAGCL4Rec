import numpy as np
from time import time, strftime, localtime
import sys
from os.path import abspath
from graph import Interaction
import math
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import json
import heapq
from graph import Data
from conf import OptionConf
from logger import Log
from os.path import abspath
from time import strftime, localtime, time
from numba import jit
from llm_reranker import LLMReranker

# 添加FileIO类的定义
class FileIO(object):
    @staticmethod
    def write_file(dir_path, file_name, content):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, file_name), 'w') as f:
            f.writelines(content)

class Recommender(object):
    def __init__(self, conf, training_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)
        self.model_name = self.config['model.name']
        self.ranking = OptionConf(self.config['item.ranking'])
        self.emb_size = int(self.config['embedding.size'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        self.batch_size = int(self.config['batch_size'])
        self.lRate = float(self.config['learnRate'])
        self.reg = float(self.config['reg.lambda'])
        self.output = OptionConf(self.config['output.setup'])
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, self.model_name + ' ' + current_time)
        self.result = []
        self.recOutput = []

    def initializing_log(self):
        self.model_log.add('### model configuration ###')
        for k in self.config.config:
            self.model_log.add(k + '=' + self.config[k])

    def print_model_info(self):
        print('Model:', self.config['model.name'])
        print('Training Set:', abspath(self.config['training.set']))
        print('Test Set:', abspath(self.config['test.set']))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:',  self.reg)
        parStr = ''
        if self.config.contain(self.config['model.name']):
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr += key[1:] + ':' + args[key] + '  '
            print('Specific parameters:', parStr)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list, raw_scores):
        pass
    
    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list, raw_scores = self.test()
        print('Evaluating...')
        self.evaluate(rec_list, raw_scores)


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        
        self.use_llm_reranker = self.config.get('use_llm_reranker', True)
        self.use_local_model = self.config.get('use_local_model', False)
        
        self.llm_reranker = None
        profile_path = '/home/zlq/submission/EMNLP2025/dataset/ml-1m/merged_profiles.json'
        
        local_model_path = self.config.get('local_model_path', '/home/zlq/submission/EMNLP2025/utils/output/qwen_reranker_lora')
        
        if self.use_llm_reranker:
            print("Initializing reranker...")
            try:
                if self.use_local_model:
                    self.llm_reranker = LLMReranker(
                        profile_path=profile_path,
                        local_model_path=local_model_path
                    )
                    print(f"Local Qwen reranker initialized using model: {local_model_path}")
                else:
                    self.llm_reranker = LLMReranker(
                        profile_path=profile_path
                    )
                    print(f"LLM reranker initialized without local model")
            except Exception as e:
                print(f"Error initializing reranker: {str(e)}")
                self.llm_reranker = None

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), int(rate * 100))
            sys.stdout.write(r)
            sys.stdout.flush()

        print("Starting prediction phase...")
        start_time = time()
        rec_list = {}
        raw_scores = {}
        user_count = len(self.data.test_set)

        all_test_users = list(self.data.test_set)
        np.random.seed(42)
        sampled_users = np.random.choice(all_test_users, size=max(1, int(0.2*len(all_test_users))), replace=False)
        user_count = len(sampled_users)
        
        print(f"Testing on {user_count} randomly sampled users (20% of {len(all_test_users)})")
        
        process_bar(0, user_count)

        for i, user in enumerate(sampled_users):
            try:
                candidates = self.predict(user)
                
                if isinstance(candidates, np.ndarray):
                    candidates = candidates.astype(np.float32)
                    scores_dict = {self.data.id2item[i]: float(score) 
                                 for i, score in enumerate(candidates)}
                    
                    rated_list, _ = self.data.user_rated(user)
                    for item in rated_list:
                        item_idx = self.data.item[item]
                        if item_idx < len(candidates):
                            candidates[item_idx] = float('-inf')

                    if self.use_llm_reranker and self.llm_reranker is not None:
                        try:
                            print(f"Reranking for user {user} using local model...")
                            candidates = self.llm_reranker.rerank(
                                candidates, 
                                user, 
                                self.data
                            )
                        except Exception as e:
                            print(f"\nReranking error for user {user}: {str(e)}")
                    
                    top_indices = np.argpartition(candidates, -self.max_N)[-self.max_N:]
                    top_indices = top_indices[np.argsort(-candidates[top_indices])]
                    rec_items = [(self.data.id2item[idx], float(candidates[idx])) 
                               for idx in top_indices]
                    
                else:
                    scores_dict = {self.data.id2item[iid]: score 
                                 for iid, score in candidates.items()}
                    
                    rated_list, _ = self.data.user_rated(user)
                    for item in rated_list:
                        if self.data.item[item] in candidates:
                            candidates[self.data.item[item]] = float('-inf')
                    
                    if self.use_llm_reranker and self.llm_reranker is not None:
                        try:
                            print(f"Reranking for user {user} using local model...")
                            candidates = self.llm_reranker.rerank(
                                candidates, 
                                user, 
                                self.data
                            )
                        except Exception as e:
                            print(f"\nReranking error for user {user}: {str(e)}")
                            
                    ids, scores = find_k_largest(self.max_N, candidates)
                    rec_items = [(self.data.id2item[idx], score) for idx, score in zip(ids, scores)]
                
                raw_scores[user] = scores_dict
                rec_list[user] = rec_items

            except Exception as e:
                print(f"\nError processing user {user}: {str(e)}")
                rec_list[user] = []
                raw_scores[user] = {}

            process_bar(i + 1, user_count)

        process_bar(user_count, user_count)
        print(f"\nPrediction completed in {time() - start_time:.2f} seconds")
        
        return rec_list, raw_scores

    def evaluate(self, rec_list, raw_scores):
        print("\nStarting evaluation phase...")
        start_time = time()
        
        self.recOutput = ['userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n']
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += f' ({item[0]},{item[1]:.4f})'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output['-dir']
        
        file_name = f"{self.config['model.name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN, raw_scores)
        
        ndcg = float(next(m.split(':')[1] for m in self.result[1:] if m.startswith('NDCG')).strip())
        
        if not self.bestPerformance or ndcg > self.bestPerformance[1]['NDCG']:
            self.bestPerformance = [self.model_name, {
                k.split(':')[0].strip(): float(k.split(':')[1]) 
                for k in self.result[1:]
            }]
        
        file_name = f"{self.config['model.name']}@{current_time}-performance.txt" 
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        
        print(f'Evaluation completed in {time() - start_time:.2f} seconds')
        print(f'Results saved to {abspath(out_dir)}')
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        
        original_use_reranker = self.use_llm_reranker
        
        if epoch >= 20 and epoch % 10 == 0:
            self.use_llm_reranker = True
            reranker_type = "local Qwen model" if self.use_local_model else "LLM without local model"
            print(f"Epoch {epoch}: Using {reranker_type} for reranking")
        else:
            self.use_llm_reranker = False
            print(f"Epoch {epoch}: NOT using reranking")
        
        rec_list, raw_scores = self.test()
        
        self.use_llm_reranker = original_use_reranker
        
        if epoch == 20:
            print("Generating cf_candidates.json file at epoch 20...")
            self._generate_cf_candidates(raw_scores)
            print("cf_candidates.json file has been generated successfully.")
        
        valid_users = {user for user in rec_list if len(rec_list[user]) > 0}
        test_subset = {user: self.data.test_set[user] for user in valid_users}
        
        measure = ranking_evaluation(test_subset, rec_list, [self.max_N], raw_scores)
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
            
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        if self.use_llm_reranker:
            reranker_type = "local Qwen model" if self.use_local_model else "LLM without local model"
            print(f"Evaluation performed on {len(valid_users)} sampled users WITH {reranker_type} reranking")
        else:
            print(f"Evaluation performed on {len(valid_users)} sampled users without reranking")
        
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        
        bp = ''
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG']) + '  |  '
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure
        
    def set_reranker_config(self, use_llm_reranker=None, use_local_model=None, local_model_path=None):
        if use_llm_reranker is not None:
            self.use_llm_reranker = use_llm_reranker
        
        if use_local_model is not None:
            self.use_local_model = use_local_model
            
        if local_model_path is not None or use_local_model is not None:
            profile_path = '/home/zlq/submission/EMNLP2025/dataset/ml-1m/merged_profiles.json'
            
            current_local_model_path = local_model_path or self.config.get('local_model_path', 
                                      '/home/zlq/submission/EMNLP2025/utils/output/qwen_reranker_lora')
            
            try:
                if self.use_local_model:
                    self.llm_reranker = LLMReranker(
                        profile_path=profile_path,
                        local_model_path=current_local_model_path
                    )
                    print(f"Reranker configuration updated to use local Qwen model")
                    print(f"Local model path: {current_local_model_path}")
                else:
                    self.llm_reranker = LLMReranker(
                        profile_path=profile_path
                    )
                    print(f"Reranker configuration updated to use LLM without local model")
            except Exception as e:
                print(f"Error updating reranker configuration: {str(e)}")
                self.llm_reranker = None

    def _generate_cf_candidates(self, raw_scores):
        cf_candidates = {}
        top_n = 100
        
        for user_id, scores in raw_scores.items():
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            top_items = [item_id for item_id, _ in sorted_items[:top_n]]
            
            cf_candidates[user_id] = top_items
        
        output_path = '/home/zlq/submission/EMNLP2025/dataset/ml-1m/cf_candidates.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cf_candidates, f, ensure_ascii=False, indent=2)
        
        print(f"CF candidates saved to {output_path} for {len(cf_candidates)} users")
        print(f"Each user has up to {top_n} candidate items")

class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall


    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)


def ranking_evaluation(origin, res, N, raw_scores=None):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
            
        print("\nCalculating basic metrics...")
        hits = Metric.hits(origin, predicted)
        
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        
            
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    k_largest_scores = [item[0] for item in n_candidates]
    return ids, k_largest_scores
