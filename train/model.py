import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import LGConv
import pdb


class MetaKRec(nn.Module):

    def __init__(self, dim, u_nodes, i_nodes, graph_num, layers, type):
        super(MetaKRec, self).__init__()
        self.layers = layers
        self.dim = dim
        self.type = type
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.node_embedding = nn.Embedding(self.i_nodes, self.dim)
        self.conv_layers = torch.nn.ModuleList([LGConv() for i in range(graph_num)])
        self.fc_cl_list = torch.nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(graph_num)])
        self.channel_linear = nn.Parameter(torch.randn(self.dim * graph_num, self.dim))

        self.W = nn.Parameter(torch.randn(self.dim, self.dim))
        self.a = nn.Parameter(torch.randn(self.dim, 1))

    def channel_attention(self, ls):
        embeddings = torch.stack(ls, dim = -2)
        tensor = torch.matmul(embeddings, self.W)
        tensor = torch.matmul(tensor, self.a)
        weight = nn.functional.softmax(tensor, dim = -2)
        embeddings = weight * embeddings
        embedding = torch.sum(embeddings, dim = -2)
        return embedding

    def channel_attention_concat(self, ls):
        embeddings = torch.concat(ls, dim = -1)
        embeddings = torch.matmul(embeddings, self.channel_linear)
        return embeddings

    def channel_attention_mean(self, ls):
        embeddings = torch.stack(ls)
        embedding = torch.mean(embeddings, dim = 0)
        return embedding

    def forward_one_graph(self, u, i, graph, layer, fc_cl):

        node_embedding = self.node_embedding(graph.x)
        for count in range(self.layers):
            node_embedding = layer(node_embedding, graph.edge_index)
        node_embedding_orig = node_embedding

        return node_embedding_orig


    def forward(self, user, item, graphs):
        ls_embedding = []

        for i in range(len(graphs)):
            embedding = self.forward_one_graph(user, item, graphs[i], self.conv_layers[i], self.fc_cl_list[i])
            ls_embedding.append(embedding)

        if self.type == 'attention':
            node_embedding = self.channel_attention(ls_embedding)
        elif self.type == 'concat':
            node_embedding = self.channel_attention_concat(ls_embedding)
        elif self.type == 'mean':
            node_embedding = self.channel_attention_mean(ls_embedding)

        u_embedding = torch.index_select(node_embedding, 0, user)
        i_embedding = torch.index_select(node_embedding, 0, item)

        out = torch.sum(u_embedding * i_embedding, dim=1)
        return out, ls_embedding


    def generate(self, graphs):
        node_embedding = self.node_embedding(graphs[0].x)

        ls_embedding = []
        for i in range(len(graphs)):
            embedding = self.conv_layers[i](node_embedding, graphs[i].edge_index)
            ls_embedding.append(embedding)

        if self.type == 'attention':
            node_embedding = self.channel_attention(ls_embedding)
        elif self.type == 'concat':
            node_embedding = self.channel_attention_concat(ls_embedding)
        elif self.type == 'mean':
            node_embedding = self.channel_attention_mean(ls_embedding)

        return node_embedding

    def predict(self, u, i, u_embedding, i_embedding):
        return torch.matmul(u_embedding, i_embedding.transpose(0, 1))

