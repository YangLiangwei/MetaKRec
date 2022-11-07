import torch
import pdb
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch_geometric.data import GraphSAINTRandomWalkSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from collections import defaultdict
from utils.parser import parse_args
from utils.utils import *
from utils import logging_setting

# def sparsify_data(train, valid, test, keep_edge):
#     train_pos_df = pd.DataFrame(data=train[train[:,2]==1], columns=['user', 'item', 'label'])
#     train_neg = train[train[:,2]==0]
#     sparse_train_pos_df = train_pos_df.groupby("item").apply(lambda x: x.sample(n=max(1, int(len(x) * keep_edge)), random_state=1234))

#     sparse_train_pos_np = sparse_train_pos_df.to_numpy()

#     sparse_train_np = np.concatenate((sparse_train_pos_np, train_neg), axis=0)

#     users_in_sparse_train = sparse_train_np[:,0]

#     sparse_valid = valid[np.isin(valid[:,0], users_in_sparse_train)]
#     sparse_test = test[np.isin(test[:,0], users_in_sparse_train)]

#     return sparse_train_np, sparse_valid, sparse_test

def sparsify_data(train, valid, test):
    train_pos_df = pd.DataFrame(data=train[train[:,2]==1], columns=['user', 'item', 'label'])
    train_neg = train[train[:,2]==0]
    sparse_train_pos_df = train_pos_df.groupby("user").apply(lambda x: x.sample(n=1, random_state=1234))

    sparse_train_pos_np = sparse_train_pos_df.to_numpy()

    sparse_train_np = np.concatenate((sparse_train_pos_np, train_neg), axis=0)

    items_in_sparse_train = sparse_train_np[:,1]

    sparse_valid = valid[np.isin(valid[:,1], items_in_sparse_train)]
    sparse_test = test[np.isin(test[:,1], items_in_sparse_train)]

    return sparse_train_np, sparse_valid, sparse_test

def log_one_data(data):
    n_users = len(np.unique(data[:, 0]))
    n_items = len(np.unique(data[:, 1]))
    interactiosns = len(data)
    #pos_interactions = interactiosns[interactiosns == 1]
    #neg_interactions = interactiosns[interactiosns == 0]
    logging.info("n_users: " + str(n_users))
    logging.info("n_items: " + str(n_items))
    logging.info("n_pos_interactions:" + str(interactiosns))
    #logging.info("n_pos_interactions: " + str(len(pos_interactions)))
    #logging.info("n_neg_interactions: " + str(len(neg_interactions)))
    #logging.info("pos/neg ratio: " + str(len(pos_interactions) / len(neg_interactions)))

args = parse_args()
logging.info(f'save debug info to {logging_setting(args)}')
logging.info('Nothing.')
numbers = get_numbers(args)
number = numbers[0]

k_list = [1, 2, 5, 10, 20, 40, 50, 60, 80, 100]
df = pd.read_csv('./data/' + args.dataset.split(',')[0] + '/ratings_final.txt',
                     sep='\t', header=None, index_col=None
                    ).values
kg_data = []
ent_set = set()
pos_triplets = defaultdict(list)
item_set = set([i for i in range(number[args.dataset.split(',')[0]]['items'])])
item_related_entities = defaultdict(list)
kg_df = pd.read_csv('./data/'
                     + args.dataset.split(',')[0] + '/kg_final.txt', sep='\t', header=None, index_col=None)
for value in kg_df.values:
    head = value[0]
    tail = value[-1]
    relation = value[1]
    kg_data.append([head, relation, tail])
    ent_set.add(head)
    ent_set.add(tail)
    pos_triplets[(head, relation)].append(tail)
    if head in item_set:
        item_related_entities[head].append(tail)
    if tail in item_set:
        item_related_entities[tail].append(head)

train, over = train_test_split(df, test_size=0.2, random_state=1234)
test, valid = train_test_split(over, test_size=0.5, random_state=1234)
sparse_train, sparse_valid, sparse_test = sparsify_data(train, valid, test)
train_hist_dict = dict()
for ind in range(train.shape[0]):
    u, i, l = train[ind][0], train[ind][1], train[ind][2]
    if l == 1:
        if u in train_hist_dict:
            train_hist_dict[u].append(i)
        else:
            train_hist_dict[u] = [i]
valid_dict = dict()
for ind in range(valid.shape[0]):
    u, i, l = valid[ind][0], valid[ind][1], valid[ind][2]
    if l == 1:
        if u in valid_dict:
            valid_dict[u].append(i)
        else:
            valid_dict[u] = [i]
test_dict = dict()
for ind in range(test.shape[0]):
    u, i, l = test[ind][0], test[ind][1], test[ind][2]
    if l == 1:
        if u in test_dict:
            test_dict[u].append(i)
        else:
            test_dict[u] = [i]
store = {}
store['train'] = train
store['test'] = test
store['valid'] = valid
#store['test_neg_items'] = test_neg_items
#store['valid_neg_items'] = valid_neg_items
store['sparse_train'] = sparse_train
store['sparse_valid'] = sparse_valid
store['sparse_test'] = sparse_test
#store['sparse_valid_neg_items'] = sparsevalid_neg_items
#store['sparse_test_neg_items'] = sparsetest_neg_items
store['name'] = args.dataset
store['number'] = numbers
store['seed'] = 1234
# pd.to_pickle(store, 'data/' + args.dataset+'/'+args.dataset+'.pkl')
# logging.info('Store the dataset:'+args.dataset)

global max_user_ind, item_enc, feat_enc, train_data, valid_data, test_data

max_user_ind = max(df[:, 0])

item_enc = OneHotEncoder().fit([[i] for i in range(number[args.dataset.split(',')[0]]['entities'])])
# if args.model.startswith('heter') or args.model.startswith('rgcn') or args.model in ['gcmc', 'lightgcn', 'fmhetergcn', 'fmhetergcn1b']:
#     item_enc = OneHotEncoder().fit([[i] for i in range(number[args.dataset]['total_entities'])])


feat_enc = MultiLabelBinarizer().fit([[i] for i in range(number[args.dataset.split(',')[0]]['entities'] + number[args.dataset.split(',')[0]]['users'])])

train_data = np.array([[u, i] for u, i, label in zip(train[:, 0], train[:, 1], train[:, 2]) if label == 1], dtype=np.int32)
valid_data = np.array([[u, i] for u, i, label in zip(valid[:, 0], valid[:, 1], valid[:, 2]) if label == 1], dtype=np.int32)
test_data = np.array([[u, i] for u, i, label in zip(test[:, 0], test[:, 1], test[:, 2]) if label == 1], dtype=np.int32)
if args.coldstartexp:
    logging.info('Cold Start Data Preparation................')
    train_data = np.array([[u, i] for u, i, label in zip(sparse_train[:, 0], sparse_train[:, 1], sparse_train[:, 2]) if label == 1], dtype=np.int32)
    valid_data = np.array([[u, i] for u, i, label in zip(sparse_valid[:, 0], sparse_valid[:, 1], sparse_valid[:, 2]) if label == 1], dtype=np.int32)
    test_data = np.array([[u, i] for u, i, label in zip(sparse_test[:, 0], sparse_test[:, 1], sparse_test[:, 2]) if label == 1], dtype=np.int32)

logging.info('Training data...')
log_one_data(train_data)
logging.info('Valid data...')
log_one_data(valid_data)
logging.info('Test data...')
log_one_data(test_data)

def gen_feed_dict_eval(start, end, data):
    feed_dict = {'users': None, 'test_items': None, 'test_feats': None}

    batch_data = data[start:end, :]
    batch_users = batch_data[:, 0]
    batch_test_items = batch_data[:, 1]
    if args.model in ['fm', 'fmkg', 'lr', 'lrkg']:
        feat_testitem_ind = []
        for u, test_item in zip(batch_users, batch_test_items):
            feat_list = [u, test_item + max_user_ind + 1]
            if 'kg' in args.model:
                for ent in item_related_entities[test_item]:
                    feat_list.append(ent + max_user_ind + 1)
            feat_testitem_ind.append(feat_list)
        feat_testitem_hot = torch.tensor(feat_enc.transform(feat_testitem_ind), dtype=torch.float)
        feed_dict['test_feats'] = feat_testitem_hot
    else:
        feed_dict['users'] = torch.tensor(batch_users, dtype=torch.long)
        feed_dict['test_items'] = torch.tensor(batch_test_items, dtype=torch.long)


    return feed_dict


def gen_feed_dict_bpr(start, end, data):
    feed_dict = {'users': None, 'pos_items': None, 'neg_items': None, 'pos_feats': None, 'neg_feats': None}

    batch_data = data[start:end, :]
    batch_users = batch_data[:, 0]
    batch_pos_items = batch_data[:, 1]
    batch_neg_items = []
    for ind in range(len(batch_users)):
        u = batch_users[ind]
        u_hist_items = set(train_hist_dict.get(u, []))
        neg_candidate = list(item_set - u_hist_items)
        sample_neg_item = np.random.choice(neg_candidate, size=1, replace=False)[0]
        batch_neg_items.append(sample_neg_item)
    batch_neg_items = np.array(batch_neg_items, dtype=np.int32)

    if args.model in ['fm', 'fmkg', 'lr', 'lrkg']:
        feat_positem_ind = []
        for u, pos_item in zip(batch_users, batch_pos_items):
            feat_list = [u, pos_item + max_user_ind + 1]
            if 'kg' in args.model:
                for ent in item_related_entities[pos_item]:
                    feat_list.append(ent + max_user_ind + 1)
            feat_positem_ind.append(feat_list)
        feat_negitem_ind = []
        for u, neg_item in zip(batch_users, batch_neg_items):
            feat_list = [u, neg_item + max_user_ind + 1]
            if 'kg' in args.model:
              for ent in item_related_entities[neg_item]:
                  feat_list.append(ent + max_user_ind + 1)
            feat_negitem_ind.append(feat_list)
        feat_positem_hot = torch.tensor(feat_enc.transform(feat_positem_ind), dtype=torch.float)
        feat_negitem_hot = torch.tensor(feat_enc.transform(feat_negitem_ind), dtype=torch.float)
        feed_dict['pos_feats'] = feat_positem_hot
        feed_dict['neg_feats'] = feat_negitem_hot
    elif args.model.startswith('heter') or args.model.startswith('rgcn') or args.model in ['gcmc', 'lightgcn', 'sgconv', 'gcn1', 'fmhetergcn', 'fmhetergcn1b']:
        feed_dict['users'] = torch.tensor(batch_users, dtype=torch.long)
        feed_dict['pos_items'] = torch.tensor(batch_pos_items + max_user_ind + 1, dtype=torch.long)
        feed_dict['neg_items'] = torch.tensor(batch_neg_items + max_user_ind + 1, dtype=torch.long)
    else:
        feed_dict['users'] = torch.tensor(batch_users, dtype=torch.long)
        feed_dict['pos_items'] = torch.tensor(batch_pos_items, dtype=torch.long)
        feed_dict['neg_items'] = torch.tensor(batch_neg_items, dtype=torch.long)

    return feed_dict



class MyDataset(Dataset):
  def __init__(self, args, mode='train', dataset='book', number=None):
    super(MyDataset, self).__init__()
    self.dataset = dataset
    self.number = number
    self.loss = args.loss
    self.model = args.model

    self.max_user_ind = max(df[:, 0])

    if args.model.startswith('heter') or args.model.startswith('rgcn') or args.model in ['gcmc', 'lightgcn', 'fmhetergcn', 'fmhetergcn1b']:
        self.item_enc = OneHotEncoder().fit([[i] for i in range(number[dataset]['total_entities'])])
    else:
        self.item_enc = OneHotEncoder().fit([[i] for i in range(number[dataset]['entities'])])

    #valid_neg_items, test_neg_items = self.get_neg_sampled_items(train, valid, test)


    self.feat_enc = MultiLabelBinarizer().fit([[i] for i in range(number[dataset]['total_entities'])])

    #sparsevalid_neg_items, sparsetest_neg_items = self.get_neg_sampled_items(sparse_train, sparse_valid, sparse_test)

    if args.coldstartexp == False:
        if mode == 'train':
            self.data = train
        elif mode == 'test':
            self.data = test
        else:
            self.data = valid
    else:
        if mode == 'train':
            self.data = sparse_train
        elif mode == 'test':
            self.data = sparse_test
        else:
            self.data = sparse_valid

    self.pos_data = self.data[self.data[:, 2]==1]
    logging.info(mode+' set size:'+str(self.data.shape[0]))
    logging.info(mode+" dataset stats: ")
    self.log_all_data(store)

  def __getitem__(self, index):
      if self.loss == 'BCE':
        temp = self.data[index]
      else:
        temp = self.pos_data[index]
      u = temp[0]
      # u_hist_items can be empty list if the dataloader is a valid or test dataloader
      u_hist_items = set(train_hist_dict.get(u, []))
      neg_candidate = list(item_set - u_hist_items)
      sample_neg_item = np.random.choice(neg_candidate, size=1, replace=False)[0]
      # one hot initialization of the item
      if self.model.startswith('heter') or self.model.startswith('rgcn') or self.model in ['gcmc', 'lightgcn', 'fmhetergcn', 'fmhetergcn1b']:
        item = self.item_enc.transform([[temp[1] + self.max_user_ind + 1]]).toarray()
        neg_item = self.item_enc.transform([[sample_neg_item + self.max_user_ind + 1]]).toarray()
      else:
        item = self.item_enc.transform([[temp[1]]]).toarray()
        neg_item = self.item_enc.transform([[sample_neg_item]]).toarray()
      user, item, label = torch.tensor(temp[0], dtype=torch.long), torch.tensor(item, dtype=torch.float), torch.tensor(
          [temp[2]], dtype=torch.double)

      neg_item = torch.tensor(neg_item, dtype=torch.float)

      if self.model.startswith('heter') or self.model.startswith('rgcn') or self.model in ['gcmc', 'lightgcn', 'fmhetergcn', 'fmhetergcn1b']:
          if index > len(kg_data) - 1:
              index = index % (len(kg_data) - 1)
          head, relation, tail = kg_data[index]
          pos_tails = set(pos_triplets[(head, relation)])
          neg_tail_candidates = list(ent_set - pos_tails)
          sample_neg_tail = np.random.choice(neg_tail_candidates, size=1, replace=False)[0]
          tail = self.item_enc.transform([[tail + self.max_user_ind + 1]]).toarray()
          neg_tail = self.item_enc.transform([[sample_neg_tail + self.max_user_ind + 1]]).toarray()
          head = torch.tensor(head, dtype=torch.long)
          tail = torch.tensor(tail, dtype=torch.float)
          neg_tail = torch.tensor(neg_tail, dtype=torch.float)
          return user, item, label, torch.tensor(temp[1], dtype=torch.long), neg_item, torch.tensor(sample_neg_item, dtype=torch.long), head, tail, neg_tail, -1, -1
      elif self.model in ['fm', 'fmkg', 'lr', 'lrkg']:
          feat_positem_ind = [u, temp[1] + self.max_user_ind + 1] # user index, item index
          feat_negitem_ind = [u, sample_neg_item + self.max_user_ind + 1]
          if 'kg' in self.model:
            for ent in item_related_entities[temp[1]]:
                feat_positem_ind.append(ent + self.max_user_ind + 1)
            for ent in item_related_entities[sample_neg_item]:
                feat_negitem_ind.append(ent + self.max_user_ind + 1)
          feat_positem_hot = torch.tensor(self.feat_enc.transform([feat_positem_ind]), dtype=torch.float)
          feat_negitem_hot = torch.tensor(self.feat_enc.transform([feat_negitem_ind]), dtype=torch.float)
          return user, item, label, torch.tensor(temp[1], dtype=torch.long), neg_item, torch.tensor(sample_neg_item, dtype=torch.long), -1, -1, -1, feat_positem_hot, feat_negitem_hot
      else:
          return user, item, label, torch.tensor(temp[1], dtype=torch.long), neg_item, torch.tensor(sample_neg_item, dtype=torch.long), -1, -1, -1, -1, -1

  def __len__(self):
      if self.loss == 'BCE':
        return len(self.data)
      else:
        return len(self.pos_data)


  def log_all_data(self, store):
    train = store["train"]
    logging.info("logging train data:")
    log_one_data(train)
    valid = store["valid"]
    logging.info("logging valid data:")
    log_one_data(valid)
    test = store["test"]
    logging.info("logging test data:")
    log_one_data(test)
    log_eval_data(train, valid, "valid")
    log_eval_data(train, test, "test")

    logging.info("logging sparse train data:")
    log_one_data(store['sparse_train'])
    logging.info("logging sparse valid data:")
    log_one_data(store['sparse_valid'])
    logging.info("logging sparse test data:")
    log_one_data(store['sparse_test'])

    log_eval_data(store['sparse_train'], store['sparse_valid'], "sparse_valid")
    log_eval_data(store['sparse_train'], store['sparse_test'], "sparse_test")

  def get_neg_sampled_items(self, train, valid, test):
    train_hist_dict = dict()
    for ind in range(train.shape[0]):
        u, i, l = train[ind][0], train[ind][1], train[ind][2]
        if l == 1:
            if u in train_hist_dict:
                train_hist_dict[u].append(i)
            else:
                train_hist_dict[u] = [i]
    test_dict = defaultdict(list)
    valid_dict = defaultdict(list)
    for ind in range(test.shape[0]):
        u, i, l = test[ind][0], test[ind][1], test[ind][2]
        if l == 1:
            test_dict[u].append(i)

    for ind in range(valid.shape[0]):
        u, i, l = valid[ind][0], valid[ind][1], valid[ind][2]
        if l == 1:
            valid_dict[u].append(i)

    test_neg_items = {}
    valid_neg_items = {}
    num_neg_items = 100
    for valid_u in valid_dict.keys():
        interacted_items = set(train_hist_dict.get(u, []) + valid_dict[valid_u])
        items_candidates = list(self.item_set - interacted_items)
        total_valid_items = list(np.random.choice(items_candidates, num_neg_items, replace=False))
        total_valid_items += valid_dict[valid_u]
        valid_neg_items[valid_u] = total_valid_items


    for test_u in test_dict.keys():
        interacted_items = set(train_hist_dict.get(u, []) + valid_dict.get(u, []) + test_dict[test_u])
        items_candidates = list(self.item_set - interacted_items)
        total_test_items = list(np.random.choice(items_candidates, num_neg_items, replace=False))
        total_test_items += test_dict[test_u]
        test_neg_items[test_u] = total_test_items

    return valid_neg_items, test_neg_items


def data_loader(number):
    train_dataset = MyDataset(args, mode='train',dataset=args.dataset, number=number)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory = True, num_workers=4)
    test_loader = DataLoader(MyDataset(args, mode='test',dataset=args.dataset, number=number), batch_size=args.batch_size, shuffle=False, pin_memory = True)
    valid_loader = DataLoader(MyDataset(args, mode='valid', dataset=args.dataset, number=number), batch_size=args.batch_size, shuffle=False, pin_memory = True)
    return train_loader, test_loader, valid_loader


def data_graphsaintloader(args, dataset, batch_size, number):
    train_dataset = MyDataset(args, mode='train',dataset=dataset, number=number)
    valid_dataset = MyDataset(args, mode='valid', dataset=dataset, number=number)
    test_dataset = MyDataset(args, mode='test', dataset=dataset, number=number)
    train_loader = GraphSAINTRandomWalkSampler(train_dataset, batch_size=batch_size, walk_length=3,
                                     num_steps=len(train_dataset), sample_coverage=20,
                                     num_workers=3, pin_memory = True)
    valid_loader = GraphSAINTRandomWalkSampler(valid_dataset, batch_size=batch_size, walk_length=3,
                                     num_steps=len(valid_dataset), sample_coverage=20,
                                     num_workers=3, pin_memory = True)
    test_loader = GraphSAINTRandomWalkSampler(test_dataset, batch_size=batch_size, walk_length=3,
                                     num_steps=len(test_dataset), sample_coverage=20,
                                     num_workers=3, pin_memory = True)
    return train_loader, test_loader, valid_loader



def log_eval_data(train, eval_data, mode):
    users_in_train = set(np.unique(train[:, 0]))
    users_in_eval = set(np.unique(eval_data[eval_data[:, 2]==1][:, 0]))

    logging.info("train and "+mode+" will eval " + str(len(users_in_train & users_in_eval)) + " users")




if __name__ == '__main__':
    DATASET = 'music'
    batch_size = 64
    number = {'music': {'users': 1872, 'items': 3846, 'interaction': 42346, 'entities': 9366, 'relations': 60, 'kg_triples': 15518, 'hyperbolicity': 0}}
    train_data, test_data, valid_data = data_loader(DATASET, batch_size, number)
    for user, item_hot, label, item in train_data:
      print(item)
