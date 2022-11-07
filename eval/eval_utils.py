import torch
from sklearn.preprocessing import OneHotEncoder

import numpy as np

########################
# topk_setting
########################
def topk_settings(train_loader, test_loader, n_item):
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 40, 50, 60, 80, 100]

    train_record = get_user_record(train_loader, True) # user_history_dict = {user: [history_item...]}
    test_record = get_user_record(test_loader, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, set(item_set), k_list

########################
# convert one-hot
########################
def convert2onehot(item_list, size):
    item = []
    item_enc = OneHotEncoder().fit([[i] for i in range(size)])
    # print('convert_onehot',item_list)
    for i in item_list:
        item_hot = item_enc.transform([[i]]).toarray()
        item.append(item_hot)
    return torch.tensor(item, dtype=torch.float)

########################
# get_user_record watched item
########################
def get_user_record(data_loader, is_train):
    user_history_dict = dict()
    for users, _, labels, items in data_loader:
        interactions = np.array([users.tolist(), items.tolist(), labels.tolist()])
        interactions = interactions.transpose()
        # print(items)
        for interaction in interactions:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if is_train or label[0] == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = set()
                user_history_dict[user].add(item)
    return user_history_dict
