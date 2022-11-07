########################
# save top@k and draw
########################
import logging
import torch
from eval.topk_eval import topk_eval
import os
import matplotlib.pyplot as plt

########################
# save model
########################
def save_model(args, model, auc):
    if not os.path.exists('../model'):
        os.mkdir('../model')
    if auc:
        model_params_file = f'../model/{args.dataset}_{args.model}_dim{args.dim}_lr.{args.lr}_weight_decay.{args.l2_weight_decay}_params_best_auc.pkl'
        torch.save(model.state_dict(), model_params_file)  # torch.save(model, '../model/gcn_' + DATASET + '.model')
        logging.info(f'Maximum AUC:{auc} Saving successfully.({model_params_file})\n\n')
    else:
        model_params_file = f'../model/{args.dataset}_{args.model}_dim{args.dim}_lr.{args.lr}_weight_decay.{args.l2_weight_decay}_params_earlystop.pkl'
        torch.save(model.state_dict(),
                   model_params_file)  # torch.save(model, '../model/gcn_' + DATASET + '.model')
        logging.info(f'Earlystop. Saving successfully.({model_params_file})\n\n')
########################
# read model
########################
def read_model(model, PATH):
    logging.info(f'Read model:{PATH}')
    model.load_state_dict(torch.load(PATH))
    model.eval()
    logging.info(model.eval())
    return model

