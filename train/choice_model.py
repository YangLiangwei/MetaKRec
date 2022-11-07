########################
# choose model
########################
import logging

import torch

from train.model import MetaKRec


def choice_model(args, u_nodes, i_nodes, device, number):

    model = MetaKRec(args.dim, u_nodes, i_nodes, args.graph_number, args.num_layers, args.type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    return model, optimizer
