# -*- coding:utf-8 -*-
# @Time: 2020/3/1 14:58
# @Author: jockwang, jockmail@126.com

# @ChTime: 2020/3/30 13:57
# @Editer: Shi, jiaqishi128@gmail.com
import torch
from torch.optim.optimizer import Optimizer

def euclidean_update(p, d_p, lr):
    p.data = p.data - lr * d_p
    return p.data

def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator

def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)

def poincare_grad(p, d_p):
    p_sqnorm = torch.clamp(torch.sum(p.data ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def poincare_update(p, d_p, lr):
    v = -lr * d_p
    p.data = full_p_exp_map(p.data, v)
    return p.data


class RiemannianSGD(Optimizer):

    def __init__(self, params, lr=0.001, param_names=[]):
        defaults = dict(lr=lr)
        super(RiemannianSGD, self).__init__(params, defaults)
        self.param_names = param_names

    def step(self, lr=None):
        loss = None
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group["lr"]
                if self.param_names[i] in ["Eh.weight", "rvh.weight"]:
                    d_p = poincare_grad(p, d_p)
                    p.data = poincare_update(p, d_p, lr)
                else:
                    p.data = euclidean_update(p, d_p, lr)
        return loss