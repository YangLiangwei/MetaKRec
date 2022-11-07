# -*- coding:utf-8 -*-
# @Time: 2020/3/18 21:39
# @Author: jockwang, jockmail@126.com
import numpy as np
import pdb
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import torch
from torch.utils.data import Dataset
import logging
import multiprocessing
import heapq
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time
from utils.mydataset import *

#cores = multiprocessing.cpu_count() // 2
cores = 4


"""
This method is for evaluating models that can only take ui pairs as inputs rather than doing cross product.
"""
def eval_topk_uipairs(model, graph, device):
    pool = multiprocessing.Pool(cores)
    user_list = list(set(train_hist_dict.keys()) & set(test_dict.keys()))

    result = {'precision': np.zeros(len(k_list)), 'recall': np.zeros(len(k_list)), 'ndcg': np.zeros(len(k_list)),
            'user_based_hit_ratio': np.zeros(len(k_list)), 'global_hit_ratio': np.zeros(len(k_list)), 'auc': 0., 'acc': 0., 'mrr': 0.}
    num_batch_users = 100

    test_items_batchusers = []
    num_items_total_list_batchusers = []
    k_lists_batchusers = []
    pos_items_list_batchusers = []
    total_pos_count = 0

    eval_ui_pairs = []
    score_maps_batchusers = {}
    all_pos_items_ranks = defaultdict(list)
    batch_users = []

    item_in_test = 0
    num_users = 0
    eval_rank_time = 0.
    eval_time = 0
    with torch.no_grad():
        model = model.eval()
        if args.model in ['fmhetergcn1b', 'fmhetergcn', 'rgcn', 'rgcnfm', 'lightgcn', 'gcn', 'sgconv', 'heterlightgcnfm']:
            node_embedding = model.generate(graph)
            u_embedding = node_embedding[:(max_user_ind+1), :]
            i_embedding = node_embedding[(max_user_ind+1):, :]
        elif args.model not in ['lr', 'lrkg', 'fm', 'fmkg', 'bprmf']:
            i_embedding = model.generate(graph)
            u_embedding = None
        else:
            i_embedding = None
            u_embedding = None
        for user in user_list:
            num_users += 1
            user_pos_items = test_dict[user]

            #items' indices here start from 0
            test_item_list = list(item_set - set(train_hist_dict[user]))

            total_pos_count += len(user_pos_items)
            for test_item in test_item_list:
                eval_ui_pairs.append([user, test_item])
            test_items_batchusers.append(test_item_list)
            num_items_total_list_batchusers.append(len(test_item_list))
            k_lists_batchusers.append(k_list)
            pos_items_list_batchusers.append(list(user_pos_items))
            batch_users.append(user)
            item_in_test += len(test_dict[user])

            if num_users % num_batch_users == 0 or (num_users == len(user_list) and len(test_items_batchusers) > 0):
                eval_ui_pairs = np.array(eval_ui_pairs, dtype=np.int32)
                eval_batch_size = args.batch_size * 5
                s = 0
                while s + eval_batch_size <= len(eval_ui_pairs) or (len(eval_ui_pairs) - s < eval_batch_size and s != len(eval_ui_pairs)):
                    end_ind = 0
                    if (len(eval_ui_pairs) - s < eval_batch_size and s != len(eval_ui_pairs)):
                        end_ind = len(eval_ui_pairs)
                    else:
                        end_ind = s + eval_batch_size
                    feed_dict = gen_feed_dict_eval(s, end_ind, eval_ui_pairs)

                    if args.model in ['fm', 'fmkg', 'lr', 'lrkg']:
                        feat_testitem_hot = feed_dict['test_feats'].to(device, non_blocking=True)
                        predict_out = model(feat_testitem_hot)
                    else:
                        users = feed_dict['users'].to(device, non_blocking=True)
                        test_items = feed_dict['test_items'].to(device, non_blocking=True)
                        predict_out = model.predict(users, test_items, u_embedding, i_embedding)

                    pred_scores = predict_out.tolist()
                    users_pred_batch = eval_ui_pairs[s:end_ind, 0]
                    test_items_pred_batch = eval_ui_pairs[s:end_ind, 1]
                    for ue, ie, score in zip(users_pred_batch.tolist(), test_items_pred_batch.tolist(), pred_scores):
                        if ue in score_maps_batchusers:
                            if type(score) == float:
                                score_maps_batchusers[ue][ie] = score
                            else:
                                score_maps_batchusers[ue][ie] = score[0]
                        else:
                            if type(score) == float:
                                score_maps_batchusers[ue] = {ie:score}
                            else:
                                score_maps_batchusers[ue] = {ie:score[0]}
                    s += eval_batch_size
                    if s > len(eval_ui_pairs):
                        s = len(eval_ui_pairs)

                batch_results_dict, batch_pos_items_ranks = get_performance_ranking(score_maps_batchusers, batch_users, test_items_batchusers, num_items_total_list_batchusers, k_lists_batchusers, pos_items_list_batchusers, pool)
                result['precision'] += batch_results_dict['precision']
                result['recall'] += batch_results_dict['recall']
                result['ndcg'] += batch_results_dict['ndcg']
                result['user_based_hit_ratio'] += batch_results_dict['user_based_hit_ratio']
                result['auc'] += batch_results_dict['auc']
                result['acc'] += batch_results_dict['acc']
                result['global_hit_ratio'] += batch_results_dict['global_hit_ratio']
                result['mrr'] += batch_results_dict['mrr']

                for batch_i, ranks_list in batch_pos_items_ranks.items():
                    all_pos_items_ranks[batch_i].extend(ranks_list)

                test_items_batchusers = []
                num_items_total_list_batchusers = []
                k_lists_batchusers = []
                pos_items_list_batchusers = []
                eval_ui_pairs = []
                score_maps_batchusers = {}
                batch_users = []

    result['recall'] /= len(user_list)
    result['precision'] /= len(user_list)
    result['ndcg'] /= len(user_list)
    result['user_based_hit_ratio'] /= len(user_list)
    result['auc'] /= len(user_list)
    result['acc'] /= len(user_list)
    result['mrr'] /= len(user_list)
    result['global_hit_ratio'] /= total_pos_count

    logging.info("item_in_test: "+str(item_in_test))

    for i in range(len(k_list)):
        k = k_list[i]
        logging.info('Mode: %s, Pre@%d:%.6f, Recall@%d:%.6f, HR@%d:%.6f, NDCG@%d:%.6f'%('Test Overall', k, result['precision'][i], k, result['recall'][i], k, result['user_based_hit_ratio'][i], k, result['ndcg'][i]))
    logging.info('Mode: %s, AUC:%.6f, ACC:%.6f, MRR:%.6f' % ('Test Overall', result['auc'], result['acc'], result['mrr']))

    items_in_intervals = number[args.dataset.split(',')[0]]['items_in_freqintervals']
    test_num_items_in_intervals = []
    interval_results = []
    for item_list in items_in_intervals:
        one_result = {'recall': np.zeros(len(k_list)), 'ndcg': np.zeros(len(k_list)), 'hit_ratio': np.zeros(len(k_list))}
        num_item_pos_interactions = 0
        all_ranks = []
        interval_items = []
        for item in item_list:
            pos_ranks_oneitem = all_pos_items_ranks.get(item, [])
            if len(pos_ranks_oneitem) > 0:
                interval_items.append(item)
            all_ranks.extend(pos_ranks_oneitem)
        for k_ind, k in enumerate(k_list):
            one_result['hit_ratio'][k_ind] = itemperf_hr(all_ranks, k)
            one_result['recall'][k_ind] = itemperf_recall(all_ranks, k)
            one_result['ndcg'][k_ind] = itemperf_ndcg(all_ranks, k, number[args.dataset.split(',')[0]]['items'])

        interval_results.append(one_result)
        test_num_items_in_intervals.append(interval_items)

    item_freq = number[args.dataset.split(',')[0]]['freq_quantiles']
    for i in range(len(item_freq)+1):
        if i == 0:
            logging.info('For items in freq between 0 - %d with %d items: ' % (item_freq[i], len(test_num_items_in_intervals[i])))
        elif i == len(item_freq):
            logging.info('For items in freq between %d - max with %d items: ' % (item_freq[i-1], len(test_num_items_in_intervals[i])))
        else:
            logging.info('For items in freq between %d - %d with %d items: ' % (item_freq[i-1], item_freq[i], len(test_num_items_in_intervals[i])))
        for k_ind in range(len(k_list)):
            k = k_list[k_ind]
            logging.info('Mode: %s, Recall@%d:%.6f, HR@%d:%.6f, NDCG@%d:%.6f' % ('Test Item Interval', k, interval_results[i]['recall'][k_ind], k, interval_results[i]['hit_ratio'][k_ind], k, interval_results[i]['ndcg'][k_ind]))

    return result


def eval_ui_dotproduct(model, graph, device):
    pool = multiprocessing.Pool(cores)
    user_list = list(set(train_hist_dict.keys()) & set(test_dict.keys()))

    result = {'precision': np.zeros(len(k_list)), 'recall': np.zeros(len(k_list)), 'ndcg': np.zeros(len(k_list)),
            'user_based_hit_ratio': np.zeros(len(k_list)), 'global_hit_ratio': np.zeros(len(k_list)), 'auc': 0., 'acc': 0., 'mrr': 0.}
    num_batch_users = 100

    test_items_batchusers = []
    num_items_total_list_batchusers = []
    k_lists_batchusers = []
    pos_items_list_batchusers = []
    total_pos_count = 0

    score_maps_batchusers = {}
    all_pos_items_ranks = defaultdict(list)
    batch_users = []

    item_in_test = 0
    num_users = 0
    eval_rank_time = 0.
    eval_time = 0

    with torch.no_grad():
        model = model.eval()
        if args.model in ['bprmf']:
            u_embedding = model.user_embedding.weight
            i_embedding = model.item_embedding.weight
        elif (args.model.startswith('hyper') and 'heter' not in args.model) or args.model in ['lightigcn']:
            u_embedding = model.user_embedding.weight
            i_embedding = model.generate(graph)
        else:
            node_embedding = model.generate(graph)
            u_embedding = node_embedding[:(max_user_ind+1), :]
            i_embedding = node_embedding[(max_user_ind+1):, :]
        s = 0
        #batch_s_t = time.time()
        all_s_t = time.time()
        while s + num_batch_users <= len(user_list) or (len(user_list) - s < num_batch_users and s != len(user_list)):
            end_ind = 0
            if (len(user_list) - s < num_batch_users and s != len(user_list)):
                end_ind = len(user_list)
            else:
                end_ind = s + num_batch_users

            batch_test_users_list = [user_list[u_ind] for u_ind in range(s, end_ind)]
            batch_test_users = torch.tensor(np.array(batch_test_users_list, dtype=np.int32), dtype=torch.long).to(device, non_blocking=True)
            batch_users_embedding = u_embedding[batch_test_users, :]

            batch_scores = model.predict(None, None, batch_users_embedding, i_embedding).cpu().data.numpy().copy()

            for u_ind, test_user in enumerate(batch_test_users_list):
                test_item_list = list(item_set - set(train_hist_dict[test_user]))
                user_pos_items = test_dict[test_user]
                item_in_test += len(test_dict[test_user])
                total_pos_count += len(user_pos_items)

                if test_user not in score_maps_batchusers:
                    score_maps_batchusers[test_user] = {}

                for each_test_item in test_item_list:
                    score_maps_batchusers[test_user][each_test_item] = batch_scores[u_ind][each_test_item]


                test_items_batchusers.append(test_item_list)
                num_items_total_list_batchusers.append(len(test_item_list))
                k_lists_batchusers.append(k_list)
                pos_items_list_batchusers.append(list(user_pos_items))
                batch_users.append(test_user)


            batch_results_dict, batch_pos_items_ranks = get_performance_ranking(score_maps_batchusers, batch_users, test_items_batchusers, num_items_total_list_batchusers, k_lists_batchusers, pos_items_list_batchusers, pool)
            result['precision'] += batch_results_dict['precision']
            result['recall'] += batch_results_dict['recall']
            result['ndcg'] += batch_results_dict['ndcg']
            result['user_based_hit_ratio'] += batch_results_dict['user_based_hit_ratio']
            result['auc'] += batch_results_dict['auc']
            result['acc'] += batch_results_dict['acc']
            result['global_hit_ratio'] += batch_results_dict['global_hit_ratio']
            result['mrr'] += batch_results_dict['mrr']

            for batch_i, ranks_list in batch_pos_items_ranks.items():
                all_pos_items_ranks[batch_i].extend(ranks_list)

            test_items_batchusers = []
            num_items_total_list_batchusers = []
            k_lists_batchusers = []
            pos_items_list_batchusers = []
            eval_ui_pairs = []
            score_maps_batchusers = {}
            batch_users = []


            s += num_batch_users
            if s > len(user_list):
                s = len(user_list)

            #batch_e_t = time.time()
            #print('one batch costs: ', (batch_e_t - batch_s_t))
            #batch_s_t = batch_e_t

    all_e_t = time.time()
    print('all eval time costs: ', (all_e_t - all_s_t))

    result['recall'] /= len(user_list)
    result['precision'] /= len(user_list)
    result['ndcg'] /= len(user_list)
    result['user_based_hit_ratio'] /= len(user_list)
    result['auc'] /= len(user_list)
    result['acc'] /= len(user_list)
    result['mrr'] /= len(user_list)
    result['global_hit_ratio'] /= total_pos_count

    logging.info("item_in_test: "+str(item_in_test))

    for i in range(len(k_list)):
        k = k_list[i]
        logging.info('Mode: %s, Pre@%d:%.6f, Recall@%d:%.6f, HR@%d:%.6f, NDCG@%d:%.6f'%('Test Overall', k, result['precision'][i], k, result['recall'][i], k, result['user_based_hit_ratio'][i], k, result['ndcg'][i]))
    logging.info('Mode: %s, AUC:%.6f, ACC:%.6f, MRR:%.6f' % ('Test Overall', result['auc'], result['acc'], result['mrr']))

    items_in_intervals = number[args.dataset.split(',')[0]]['items_in_freqintervals']
    test_num_items_in_intervals = []
    interval_results = []
    for item_list in items_in_intervals:
        one_result = {'recall': np.zeros(len(k_list)), 'ndcg': np.zeros(len(k_list)), 'hit_ratio': np.zeros(len(k_list))}
        num_item_pos_interactions = 0
        all_ranks = []
        interval_items = []
        for item in item_list:
            pos_ranks_oneitem = all_pos_items_ranks.get(item, [])
            if len(pos_ranks_oneitem) > 0:
                interval_items.append(item)
            all_ranks.extend(pos_ranks_oneitem)
        for k_ind, k in enumerate(k_list):
            one_result['hit_ratio'][k_ind] = itemperf_hr(all_ranks, k)
            one_result['recall'][k_ind] = itemperf_recall(all_ranks, k)
            one_result['ndcg'][k_ind] = itemperf_ndcg(all_ranks, k, number[args.dataset.split(',')[0]]['items'])

        interval_results.append(one_result)
        test_num_items_in_intervals.append(interval_items)

    item_freq = number[args.dataset.split(',')[0]]['freq_quantiles']
    for i in range(len(item_freq)+1):
        if i == 0:
            logging.info('For items in freq between 0 - %d with %d items: ' % (item_freq[i], len(test_num_items_in_intervals[i])))
        elif i == len(item_freq):
            logging.info('For items in freq between %d - max with %d items: ' % (item_freq[i-1], len(test_num_items_in_intervals[i])))
        else:
            logging.info('For items in freq between %d - %d with %d items: ' % (item_freq[i-1], item_freq[i], len(test_num_items_in_intervals[i])))
        for k_ind in range(len(k_list)):
            k = k_list[k_ind]
            logging.info('Mode: %s, Recall@%d:%.6f, HR@%d:%.6f, NDCG@%d:%.6f' % ('Test Item Interval', k, interval_results[i]['recall'][k_ind], k, interval_results[i]['hit_ratio'][k_ind], k, interval_results[i]['ndcg'][k_ind]))

    return result



def get_performance_ranking(score_maps_batchusers, batch_users, test_items_batchusers, num_items_total_list_batchusers, k_lists_batchusers, pos_items_list_batchusers, pool):
    result = {'precision': np.zeros(len(k_list)), 'recall': np.zeros(len(k_list)), 'ndcg': np.zeros(len(k_list)),
            'user_based_hit_ratio': np.zeros(len(k_list)), 'global_hit_ratio': np.zeros(len(k_list)), 'auc': 0., 'acc': 0., 'mrr': 0.}
    rating_map_list = []
    for user in batch_users:
        rating_map_list.append(score_maps_batchusers[user])

    user_batch_rating_uid = zip(rating_map_list, test_items_batchusers, num_items_total_list_batchusers, k_lists_batchusers, pos_items_list_batchusers)
    batch_result = pool.map(test_one_user, user_batch_rating_uid)

    batch_pos_items_ranks = defaultdict(list)
    for oneresult in batch_result:
        re = oneresult[0]
        pos_items_ranks = oneresult[1]
        for i, rank_list in pos_items_ranks.items():
            batch_pos_items_ranks[i].extend(rank_list)

        result['precision'] += re['precision']
        result['recall'] += re['recall']
        result['ndcg'] += re['ndcg']
        result['user_based_hit_ratio'] += re['user_based_hit_ratio']
        result['auc'] += re['auc']
        result['acc'] += re['acc']
        result['global_hit_ratio'] += re['global_hit_ratio']
        result['mrr'] += re['mrr']

    return result, batch_pos_items_ranks



class TopK():
    def __init__(self, train_loader, test_loader, number, args, graph, device):
        self.k_list = k_list
        self.item_set = item_set
        #self.user_list, self.train_record, self.test_record = \
        #    self.topk_settings(train_loader, test_loader, number, args)
        self.user_list = list(set(train_hist_dict.keys()) & set(test_dict.keys()))
        if args.model.startswith("heter") or args.model.startswith("rgcn") or args.model in ['fm', 'fmkg', 'lr', 'lrkg', 'gcmc', 'lightgcn', 'fmhetergcn', 'fmhetergcn1b']:
            n_item = number[args.dataset]['total_entities']
        else:
            n_item = number[args.dataset]['entities']
        if args.model in ['fm', 'fmkg', 'lr', 'lrkg']:
            self.item_enc = MultiLabelBinarizer().fit([[i] for i in range(n_item)])
        else:
            self.item_enc = OneHotEncoder().fit([[i] for i in range(n_item)])


        self.batch_size = args.batch_size
        self.device = device
        self.graph = graph
        self.model = args.model
        self.max_user_ind = number[args.dataset]['users']
        self.number = number
        self.dataset = args.dataset
        self.coldstartexp = args.coldstartexp

        #store = pd.read_pickle('data/' + self.dataset+'/'+self.dataset+'.pkl')

        #if args.coldstartexp:
        #    self.valid_eval_items = store['sparse_valid_neg_items']
        #    self.test_eval_items = store['sparse_test_neg_items']
        #else:
        #    self.valid_eval_items = store['valid_neg_items']
        #    self.test_eval_items = store['test_neg_items']


    def get_user_record(self, data_loader, is_train):
        user_history_dict = dict()
        for users, _, labels, items, _, _, _, _, _, _, _ in data_loader:
            interactions = np.array([users.tolist(), items.tolist(), labels.tolist()])
            interactions = interactions.transpose()
            for interaction in interactions:
                user = interaction[0]
                item = interaction[1]
                label = interaction[2]
                if is_train or label[0] == 1:
                    if user not in user_history_dict:
                        user_history_dict[user] = set()
                    user_history_dict[user].add(item)
        return user_history_dict

    def topk_settings(self, train_loader, test_loader, number, args):
        #user_num = 100

        train_record = self.get_user_record(train_loader, True)
        test_record = self.get_user_record(test_loader, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        #if len(user_list) > user_num:
        #    user_list = np.random.choice(user_list, size=user_num, replace=False)
        print("#evaluating users is: ", len(user_list))
        #item_set = []
        #for i in test_record.items():
        #    item_set.extend(i[1])
        return user_list, train_record, test_record

    def eval(self, model, path, mode):
        #num_neg_sample_items = 100
        pool = multiprocessing.Pool(cores)

        result = {'precision': np.zeros(len(self.k_list)), 'recall': np.zeros(len(self.k_list)), 'ndcg': np.zeros(len(self.k_list)),
            'user_based_hit_ratio': np.zeros(len(self.k_list)), 'global_hit_ratio': np.zeros(len(self.k_list)), 'auc': 0., 'acc': 0.}

        #precision_list = {k: [] for k in self.k_list}
        #recall_list = {k: [] for k in self.k_list}
        #hr_list = {k: [] for k in self.k_list}
        #ndcg_list = {k: [] for k in self.k_list}
        #auc_list = []

        num_batch_users = 100

        test_items_batchusers = []
        num_items_total_list_batchusers = []
        k_lists_batchusers = []
        pos_items_list_batchusers = []
        batchusers = []
        total_pos_count = 0

        eval_ui_pairs = []
        score_maps_batchusers = {}

        pos_items_pred_rank = defaultdict(list)

        #result = open(path.replace('./model/', './recommendation/')+mode+'_recommendation.txt', 'w')
        item_in_test = 0
        num_users = 0
        eval_rank_time = 0.
        eval_time = 0
        for user in self.user_list:
            num_users += 1
            #test_item_list = list(self.item_set-self.train_record[user])
            user_pos_items = test_dict[user]
            neg_candidate_items = list(self.item_set - set(train_hist_dict[user]) - set(test_dict[user]))
            #test_neg_items = np.random.choice(neg_candidate_items, size=num_neg_sample_items, replace=False)
            test_item_list = neg_candidate_items + list(user_pos_items)
            #if mode == "valid":
            #    test_item_list = self.valid_eval_items[user]
            #else:
            #    test_item_list = self.test_eval_items[user]

            total_pos_count += len(user_pos_items)
            for test_item in test_item_list:
                eval_ui_pairs.append((user, test_item))
            test_items_batchusers.append(test_item_list)
            num_items_total_list_batchusers.append(len(test_item_list) - len(user_pos_items))
            k_lists_batchusers.append(self.k_list)
            pos_items_list_batchusers.append(list(user_pos_items))
            batch_users.append(user)

            #item_score_map = dict()
            item_in_test += len(test_dict[user])
            #dataloader = torch.utils.data.DataLoader(dataset=miniDataset(user, test_item_list, self.item_enc),
            #                                         batch_size=self.batch_size, shuffle=False)
            #for _, [u, i, item] in enumerate(dataloader):
            #    u, i = u.to(self.device), i.to(self.device)
            #    outs = model(u, i, self.graph).tolist()
            #    for ie, score in zip(item.tolist(), outs):
            #        item_score_map[ie] = score

            if num_users % num_batch_users == 0 or (num_users == len(self.user_list) and len(test_items_batchusers) > 0):
                eval_dataloader = torch.utils.data.DataLoader(dataset=evalDataset(eval_ui_pairs, self.item_enc, self.model, self.max_user_ind, item_related_entities),
                                                    batch_size=self.batch_size, shuffle=False, pin_memory=True)

                for _, [u, i, user, item, relation] in enumerate(eval_dataloader):
                    eval_batch_start_time = time.time()
                    if (self.model.startswith("fm") and self.model not in ["fm", "fmkg", 'fmhetergcn', 'fmhetergcn1b']) or self.model in ["bprmf"]:
                        u, i = u.to(self.device, non_blocking=True), item.to(self.device, non_blocking=True)
                    else:
                        u, i = u.to(self.device, non_blocking=True), i.to(self.device, non_blocking=True)
                    if self.model in ["fm", "fmkg", "lr", "lrkg"]:
                        outs = model(i).tolist()
                    else:
                        outs = model(u, i, self.graph).tolist()
                    for ue, ie, score in zip(user.tolist(), item.tolist(), outs):
                        if ue in score_maps_batchusers:
                            if type(score) == float:
                                score_maps_batchusers[ue][ie] = score
                            else:
                                score_maps_batchusers[ue][ie] = score[0]
                        else:
                            if type(score) == float:
                                score_maps_batchusers[ue] = {ie:score}
                            else:
                                score_maps_batchusers[ue] = {ie:score[0]}
                    eval_batch_end_time = time.time()
                    eval_time += eval_batch_end_time - eval_batch_start_time


                ranking_start_time = time.time()
                rating_map_list = []
                for user in batch_users:
                    rating_map_list.append(score_maps_batchusers[user])

            #sorted_items = sorted(item_score_map.items(), key = lambda x: x[1], reverse=True)
            #items_sorted = [s[0] for s in sorted_items]
            #line = 'user:' + str(user) + ' ' + ' '.join(list(map(str, items_sorted[:num_neg_sample_items])))+'\n'
            #result.write(line)
            #rel = []
            #for i in items_sorted[:max(self.k_list)]:
            #    if i in self.test_record[user]:
            #        rel.append(1)
            #    else:
            #        rel.append(0)
            #rel = rank_corrected(np.array(rel), len(test_item_list), len(neg_candidate_items))
            #print(len(test_item_list), len(neg_candidate_items))
            #for k in self.k_list:
            #    #hit_num = len(set(items_sorted[:k]) & self.test_record[user])
            #    #precision_list[k].append(hit_num / k)
            #    #recall_list[k].append(hit_num / len(self.test_record[user]))
            #    #hr_list[k].append(hit_num)
            #    hr_list[k].append(hit_at_k(rel, k))
            #    recall_list[k].append(recall_at_k(rel, k, len(self.test_record[user])))
            #    precision_list[k].append(precision_at_k(rel, k))
            #    ndcg_list[k].append(ndcg_at_k(rel, k))

            ##ground_truth_score = np.zeros(len(outs))
            ##ground_truth_score[:len(test_neg_items)] = 1.0
            ##auc_list.append(roc_auc_score(ground_truth_score, outs))
        #result.close()
                user_batch_rating_uid = zip(rating_map_list, test_items_batchusers, num_items_total_list_batchusers, k_lists_batchusers, pos_items_list_batchusers)
                batch_result = pool.map(test_one_user, user_batch_rating_uid)


                all_pos_items_ranks = defaultdict(list)
                for oneresult in batch_result:
                    re = oneresult[0]
                    pos_items_ranks = oneresult[1]
                    for i, rank_list in pos_items_ranks.items():
                        all_pos_items_ranks[i].extend(rank_list)

                    result['precision'] += re['precision']/len(self.user_list)
                    result['recall'] += re['recall']/len(self.user_list)
                    result['ndcg'] += re['ndcg']/len(self.user_list)
                    result['user_based_hit_ratio'] += re['user_based_hit_ratio']/len(self.user_list)
                    result['auc'] += re['auc']/len(self.user_list)
                    result['acc'] += re['acc']/len(self.user_list)
                    result['global_hit_ratio'] += re['global_hit_ratio']


                test_items_batchusers = []
                num_items_total_list_batchusers = []
                k_lists_batchusers = []
                pos_items_list_batchusers = []
                eval_ui_pairs = []
                score_maps_batchusers = {}
                batch_users = []
                ranking_end_time = time.time()
                eval_rank_time += (ranking_end_time - ranking_start_time)
        #precision = [np.mean(precision_list[k]) for k in self.k_list]
        #recall = [np.mean(recall_list[k]) for k in self.k_list]
        #hr = [np.sum(hr_list[k]) for k in self.k_list]
        #ndcg = [np.mean(ndcg_list[k]) for k in self.k_list]

        result['global_hit_ratio'] /= total_pos_count

        print("Evaluation takes time: ", eval_time + eval_rank_time)

        logging.info("item_in_test: "+str(item_in_test))

        for i in range(len(self.k_list)):
            k = self.k_list[i]
            logging.info('Mode: %s, Pre@%d:%.6f, Recall@%d:%.6f, HR@%d:%.6f, NDCG@%d:%.6f'%(mode, k, result['precision'][i], k, result['recall'][i], k, result['user_based_hit_ratio'][i], k, result['ndcg'][i]))


        items_in_intervals = self.number[self.dataset]['items_in_freqintervals']
        test_num_items_in_intervals = []
        interval_results = []
        for item_list in items_in_intervals:
            one_result = {'recall': np.zeros(len(self.k_list)), 'ndcg': np.zeros(len(self.k_list)), 'hit_ratio': np.zeros(len(self.k_list))}
            num_item_pos_interactions = 0
            all_ranks = []
            interval_items = []
            for item in item_list:
                pos_ranks_oneitem = all_pos_items_ranks.get(item, [])
                if len(pos_ranks_oneitem) > 0:
                    interval_items.append(item)
                all_ranks.extend(pos_ranks_oneitem)
            for k_ind, k in enumerate(self.k_list):
                one_result['hit_ratio'][k_ind] = itemperf_hr(all_ranks, k)
                one_result['recall'][k_ind] = itemperf_recall(all_ranks, k)
                one_result['ndcg'][k_ind] = itemperf_ndcg(all_ranks, k, self.number[self.dataset]['items'])

            interval_results.append(one_result)
            test_num_items_in_intervals.append(interval_items)

        item_freq = self.number[self.dataset]['freq_quantiles']
        for i in range(len(item_freq)+1):
            if i == 0:
                logging.info('For items in freq between 0 - %d with %d items: ' % (item_freq[i], len(test_num_items_in_intervals[i])))
            elif i == len(item_freq):
                logging.info('For items in freq between %d - max with %d items: ' % (item_freq[i-1], len(test_num_items_in_intervals[i])))
            else:
                logging.info('For items in freq between %d - %d with %d items: ' % (item_freq[i-1], item_freq[i], len(test_num_items_in_intervals[i])))
            for k_ind in range(len(self.k_list)):
                k = self.k_list[k_ind]
                logging.info('Mode: %s, Recall@%d:%.6f, HR@%d:%.6f, NDCG@%d:%.6f'%(mode, k, interval_results[i]['recall'][k_ind], k, interval_results[i]['hit_ratio'][k_ind], k, interval_results[i]['ndcg'][k_ind]))




class miniDataset(Dataset):
    def __init__(self, user, test_item_list, item_enc):
        super(miniDataset, self).__init__()

        self.item_enc = item_enc
        self.user = user
        self.test_item_list = test_item_list

    def __getitem__(self, index):
        item = self.item_enc.transform([[self.test_item_list[index]]]).toarray()
        return torch.tensor(self.user, dtype=torch.long), torch.tensor(item, dtype=torch.float), torch.tensor(
            self.test_item_list[index], dtype=torch.long)

    def __len__(self):
        return len(self.test_item_list)


class evalDataset(Dataset):
    def __init__(self, ui_pairs, item_enc, model, max_user_ind, item_related_entities):
        super(evalDataset, self).__init__()

        self.item_enc = item_enc
        self.ui_pairs = ui_pairs
        self.model = model
        self.max_user_ind = max_user_ind
        self.item_related_entities = item_related_entities

    def __getitem__(self, index):
        user, item = self.ui_pairs[index]
        if self.model == "hetergcn" or self.model.startswith("rgcn") or self.model in ['gcmc', 'lightgcn', 'fmhetergcn', 'fmhetergcn1b']:
            item_onehot = self.item_enc.transform([[item + self.max_user_ind]]).toarray()
        elif self.model in ['fm', 'fmkg', 'lr', 'lrkg']:
            feat_item_ind = [user, item + self.max_user_ind] # user index, item index
            if 'kg' in self.model:
              for ent in self.item_related_entities[item]:
                  feat_item_ind.append(ent + self.max_user_ind)
            item_onehot = self.item_enc.transform([feat_item_ind])
        else:
            item_onehot = self.item_enc.transform([[item]]).toarray()
        relation = 0
        return torch.tensor(user, dtype=torch.long), torch.tensor(item_onehot, dtype=torch.float), torch.tensor(
            user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(relation, dtype=torch.long)

    def __len__(self):
        return len(self.ui_pairs)


def ranklist_by_heapq(item_score, user_pos_test, test_items, Ks, num_items_total):
    K_max = max(Ks)

    ranked_item_scores = sorted(item_score.items(), key=lambda x:x[1], reverse=True)
    pos_ranks = defaultdict(list)
    for ind in range(len(ranked_item_scores)):
        pred_item = ranked_item_scores[ind][0]
        if pred_item in user_pos_test:
            pos_ranks[pred_item].append(ind+1)
    K_max_item_score = [onescore[0] for onescore in ranked_item_scores[:K_max]]

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    #r = rank_corrected(np.array(r), len(test_items), num_items_total)
    auc = get_auc(ranked_item_scores, user_pos_test)
    acc = get_acc(ranked_item_scores, user_pos_test)

    return r, auc, acc, pos_ranks


def get_auc(item_score, user_pos_test):
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = cal_auc(ground_truth=r, prediction=posterior)
    return auc

def get_acc(item_score, user_pos_test):
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    acc = cal_acc(ground_truth=r, prediction=posterior)
    return acc

def get_performance(user_pos_test, r, auc, acc, Ks):
    precision, recall, ndcg, hit_ratio, hit_num = [], [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))
        hit_num.append(np.sum(np.asfarray(r)[:K]))
    mrr = cal_mrr(r)

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'user_based_hit_ratio': np.array(hit_ratio), 'global_hit_ratio': np.array(hit_num), 'auc': auc, 'acc': acc, 'mrr': mrr}

def test_one_user(x):
    # user u's ratings for user u
    rating_map = x[0]
    test_items = x[1]
    num_items_total = x[2]
    Ks = x[3]

    user_pos_test = x[4]


    r, auc, acc, pos_ranks = ranklist_by_heapq(rating_map, user_pos_test, test_items, Ks, num_items_total)

    return (get_performance(user_pos_test, r, auc, acc, Ks), pos_ranks)

def cal_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def cal_acc(ground_truth, prediction):
    temp = [0 if i < 0.5 else 1 for i in prediction]
    acc = accuracy_score(y_true=ground_truth, y_pred=temp)
    return acc

def get_ndcg(pred_rel):
    dcg = 0
    for (index,rel) in enumerate(pred_rel):
        dcg += (rel * np.reciprocal(np.log2(index+2)))
    # print("dcg " + str(dcg))
    idcg = 0
    for(index,rel) in enumerate(sorted(pred_rel,reverse=True)):
        idcg += (rel * np.reciprocal(np.log2(index+2)))
    # print("idcg " + str(idcg))
    if idcg == 0.0:
        ndcg = 0
    else:
        ndcg = dcg/idcg
    return ndcg

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1:
            continue
        corrected_r[corrected_rank] = 1
    return corrected_r


def itemperf_hr(ranks, k):
    ranks = np.array(ranks)
    if len(ranks) == 0.0:
        return 0
    return np.sum(ranks<=k) / (k * len(ranks))

def itemperf_recall(ranks, k):
    ranks = np.array(ranks)
    if len(ranks) == 0:
        return 0
    return np.sum(ranks<=k) / len(ranks)

def itemperf_ndcg(ranks, k, size):
    ndcg = 0.0
    for onerank in ranks:
        r = np.zeros(size)
        r[onerank-1] = 1
        ndcg += ndcg_at_k(r, k)
    if len(ranks) == 0:
        return 0
    return ndcg / len(ranks)

def cal_mrr(r):
    r = np.array(r)
    if np.sum(r) == 0:
        return 0.
    else:
        return np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
