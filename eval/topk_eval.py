''' TOP@K Evaluation '''
import logging
from tqdm import tqdm
import torch
import numpy as np

from eval.eval_utils import convert2onehot
########################
# top@k hr@k evaluation
########################
def topk_eval(device, model, graph, user_list, train_record, test_record, item_set, i_nodes, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    hr_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    item_in_test = 0 # compute hr

    result = open('./result.txt','w')

    #TODO
    logging.info('Evaluation:TOP@K...')
    for user in tqdm(user_list):
        item_score_map = dict()  # item_n --> score_n
        test_item_list = list(item_set - train_record[user])
        # hr count
        item_in_test += len(test_record[user])
        start = 0
        while start + batch_size <= len(test_item_list):
            items = test_item_list[start:start + batch_size]
            u = torch.tensor([user]*batch_size, dtype=torch.long).to(device)
            i = convert2onehot(items, i_nodes).to(device)
            scores = model(u, i, graph).tolist()
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size
        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            u = torch.tensor([user]*batch_size, dtype=torch.long).to(device)
            items = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
            i = convert2onehot(items, i_nodes).to(device)
            scores = model(u, i, graph).tolist()
            # item_n --> score_n
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        line = 'user:'+str(user)+' '+' '.join(item_sorted[:100])
        # logging.info(line)
        result.write(line)

        # topk
        for k in k_list:
            # hit
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            # precision
            precision_list[k].append(hit_num / k)
            # recall
            recall_list[k].append(hit_num / len(test_record[user]))
            # topk中在测试集中的item数量 / 所有用户预测item在测试集中的数量
            hr_list[k].append(hit_num)
            # ndcg
            # TODO
            ndcg_rel = []
            for i in item_sorted[:k]:
                if i in test_record[user]:
                    ndcg_rel.append(1)
                else:
                    ndcg_rel.append(0)
            ndcg_list[k].append(get_ndcg(ndcg_rel))

    result.close()
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    # f1 = [(2*precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(k_list))]
    f1 = []
    for i in range(len(k_list)):
        if precision[i]== 0 and recall[i] == 0:
            f1.append(0.0)
        else:
            f1.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))

    hr = [np.sum(hr_list[k])/item_in_test for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]
    for i in range(len(k_list)):
        k = k_list[i]
        logging.info(f'Precision@{k:<3}:{precision[i]:.6f}, Recall@{k:<3}:{recall[i]:.6f}, F-1@{k:<3}:{f1[i]:.6f}, HR@{k:<3}:{hr[i]:.6f}, NDCG@{k:<3}:{ndcg[i]:.6f}')

    return f1, precision, recall, hr, ndcg

def new_topk_eval(device, model, graph, user_list, record, item_set, i_nodes, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    hr_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    item_in_test = 0 # compute hr

    #TODO
    logging.info('Evaluation:TOP@K...')
    for user in tqdm(user_list):
        item_score_map = dict()  # item_n --> score_n
        test_item_list = list(item_set - record[user]['train'])
        # hr count
        item_in_test += len(record[user]['test'])
        start = 0
        while start + batch_size <= len(test_item_list):
            items = test_item_list[start:start + batch_size]
            u = torch.tensor([user]*batch_size, dtype=torch.long).to(device)
            i = convert2onehot(items, i_nodes).to(device)
            scores = model(u, i, graph).tolist()
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size
        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            u = torch.tensor([user]*batch_size, dtype=torch.long).to(device)
            items = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
            i = convert2onehot(items, i_nodes).to(device)
            scores = model(u, i, graph).tolist()
            # item_n --> score_n
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = list(set([i[0] for i in item_score_pair_sorted]))
        # logging.info(str(item_score_pair_sorted))
        # topk
        for k in k_list:
            # hit
            hit_num = len(set(item_sorted[:k]) & record[user]['test'])
            # precision
            precision_list[k].append(hit_num / k)
            # recall
            recall_list[k].append(hit_num / len(record[user]['test']))
            # topk中在测试集中的item数量 / 所有用户预测item在测试集中的数量
            hr_list[k].append(hit_num)
            # ndcg
            # TODO
            ndcg_rel = []
            for i in item_sorted[:k]:
                if i in record[user]['test']:
                    ndcg_rel.append(1)
                else:
                    ndcg_rel.append(0)
            ndcg_list[k].append(get_ndcg(ndcg_rel))
            # logging.info('k=' + str(k) + ' precision_list:' + str(precision_list[k]))
            # logging.info('k=' + str(k) + ' recall list:' + str(recall_list[k]))
            # logging.info('k=' + str(k) + ' hr list:' + str(hr_list[k]))
            # logging.info('k=' + str(k) + ' ndcg_rel:' + str(ndcg_rel))
            # logging.info('k=' + str(k) + ' ndcg:' + str(ndcg_list[k]) + '\n')

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    # f1 = [(2*precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(k_list))]
    f1 = []
    for i in range(len(k_list)):
        if precision[i]== 0 and recall[i] == 0:
            f1.append(0.0)
        else:
            f1.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))

    hr = [np.sum(hr_list[k])/item_in_test for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]
    for i in range(len(k_list)):
        k = k_list[i]
        logging.info(f'Precision@{k:<3}:{precision[i]:.6f}, Recall@{k:<3}:{recall[i]:.6f}, F-1@{k:<3}:{f1[i]:.6f}, HR@{k:<3}:{hr[i]:.6f}, NDCG@{k:<3}:{ndcg[i]:.6f}')
    # logging.info('precision:' + str(precision))
    # logging.info('recall:' + str(recall))
    # logging.info('f1:' + str(f1))
    # logging.info('hr:' + str(hr))
    # logging.info('ndcg:' + str(ndcg) + '\n\n\n')
    return f1, precision, recall, hr, ndcg

########################
# ndcg@k function
########################
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
