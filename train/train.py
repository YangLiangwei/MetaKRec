import time
import logging
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools

from eval.topk import eval_ui_dotproduct
from train.choice_model import choice_model
from train.train_utils import *

from utils.EarlyStoppingCriterion import *
from utils.mydataset import *
import torch.nn.functional as F


########################
# BPR Loss
########################
def bpr_loss(pos_score, neg_score):
    loss = -((pos_score - neg_score).sigmoid().log().mean())
    return loss

def eval_loss(data, model, device, graphs):

    s = 0
    eval_losses = []
    with torch.no_grad():
        model = model.eval()
        while s + args.batch_size <= len(data) or (len(data) - s < args.batch_size and s != len(data)):
            if (len(data) - s < args.batch_size and s != len(data)):
                feed_dict = gen_feed_dict_bpr(s, len(data), data)
            else:
                feed_dict = gen_feed_dict_bpr(s, s + args.batch_size, data)

            users = feed_dict['users'].to(device, non_blocking=True)
            pos_items = feed_dict['pos_items'].to(device, non_blocking=True)
            neg_items = feed_dict['neg_items'].to(device, non_blocking=True)

            pos_ui_out, _ = model(users, pos_items, graphs)
            neg_ui_out, _ = model(users, neg_items, graphs)
            pos_ui_out = pos_ui_out.double()
            neg_ui_out = neg_ui_out.double()

            loss = bpr_loss(pos_ui_out, neg_ui_out)

            eval_losses.append(loss.item())
            s += args.batch_size
            if s > len(data):
                s = len(data)
        return eval_losses


def train(train_data, valid_data, test_data):
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpu))
    cur_gpu_info = gpu_empty_info()
    logging.info('Current %s' % cur_gpu_info[0])
    logging.info('Use device:'+str(device))
    # Early stop and checkpoint
    model_params_file =f'./model/{args.dataset}_{args.model}_num_layers.{args.num_layers}_dim{args.dim}_lr.{args.lr}_weight_decay.{args.l2_weight_decay}.pkl'
    early_stopping = EarlyStoppingCriterion(patience=args.early_stop_patience,save_path=model_params_file) ## initialize the early_stopping object
    stop_epoch, auc_max = 0, 0.0

    # graph
    graphs = []
    for i in range(len(args.dataset.split(','))):
        graph = get_heter_graph(args.dataset.split(',')[i], numbers, args.model, args.coldstartexp).to(device)
        graphs.append(graph)

    u_nodes, i_nodes = number[args.dataset.split(',')[0]]['users'], number[args.dataset.split(',')[0]]['total_entities']
    # u_nodes, i_nodes = number[args.dataset.split(',')[0]]['users'], number[args.dataset.split(',')[0]]['entities']

    args.graph_number = len(args.dataset.split(','))
    model, optimizer = choice_model(args, u_nodes, i_nodes, device, number)


    logging.info(model)
    logging.info(optimizer)

    start_total_time = time.time()
    each_epoch_time = []

    all_train_losses = []
    all_valid_losses = []
    all_test_losses = []
    train_total_time = 0
    for epoch in range(args.epochs):
        train_losses = []
        valid_losses = []
        test_losses = []
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        s = 0
        train_time = 0
        batch_start_time = time.time()
        model = model.train()
        while s + args.batch_size <= len(train_data):
            feed_dict = gen_feed_dict_bpr(s, s + args.batch_size, train_data)
            optimizer.zero_grad()

            users = feed_dict['users'].to(device, non_blocking=True)
            pos_items = feed_dict['pos_items'].to(device, non_blocking=True)
            neg_items = feed_dict['neg_items'].to(device, non_blocking=True)

            pos_ui_out, ls_embedding = model(users, pos_items, graphs)
            pos_ui_out = pos_ui_out.double()

            neg_ui_out, _ = model(users, neg_items, graphs)
            neg_ui_out = neg_ui_out.double()

            loss = bpr_loss(pos_ui_out, neg_ui_out)

            loss.backward()
            optimizer.step()
            batch_end_time = time.time()
            train_time += batch_end_time - batch_start_time
            batch_start_time = batch_end_time
            train_losses.append(loss.item())
            s += args.batch_size
        print("Training time in this epoch: ", train_time)

        train_total_time += train_time
        all_train_losses.append(np.mean(train_losses))
        valid_losses = eval_loss(valid_data, model, device, graphs)
        test_losses = eval_loss(test_data, model, device, graphs)
        all_valid_losses.append(np.mean(valid_losses))
        all_test_losses.append(np.mean(test_losses))
        logging.info('epoch '+str(epoch+1)+': Train loss:%.6f Valid loss:%.6f Test loss:%.6f'%(np.mean(train_losses), np.mean(valid_losses), np.mean(test_losses)))
        early_stop_loss = np.mean(valid_losses)
        early_stopping(early_stop_loss, model, model_params_file)
        if early_stopping.early_stop:
            logging.info("Early stopping. Epochs:%d early_stop_loss:%.6f" % (epoch + 1, early_stop_loss))
            break

    print("Overall Training time is: ", train_total_time)



    logging.info('Load best Model:')
    model = read_model(model, model_params_file)

    test_results = eval_ui_dotproduct(model, graphs, device)
    print(test_results)

########################
# main function
########################
def main():
    logging.info(f'save debug info to {logging_setting(args)}')
    logging.info(args)
    torch.autograd.set_detect_anomaly(True)

    if not number:
        return
    train(train_data, valid_data, test_data)

