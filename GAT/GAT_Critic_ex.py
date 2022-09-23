# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import numpy as np
#
# class GRUGate(nn.Module):
#     def __init__(self, d_model):
#         super(GRUGate,self).__init__()
#
#         self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
#         self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
#         self.linear_w_z = nn.Linear(d_model, d_model)
#         self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
#         self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
#         self.linear_u_g = nn.Linear(d_model, d_model, bias=False)
#
#         self.init_bias()
#
#     def init_bias(self):
#         with torch.no_grad():
#             self.linear_w_z.bias.fill_(-2)
#
#     def forward(self, x, y):
#         z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))
#         r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
#         h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r*x))
#         return (1.-z)*x + z*h_hat
#
# class GraphAttentionLayer(nn.Module):
#     """GAT layer, refer to https://arxiv.org/abs/1710.10903"""
#     def __init__(self,in_features,out_features,args, activation=nn.ELU()):
#         super(GraphAttentionLayer, self).__init__()
#         self.args=args
#         self.concat = args.concat
#         self.in_features = in_features
#         self.out_features = out_features
#         self.head_num = args.GAT_head_num
#         self.skip_residual = args.GAT_skip_Residual
#
#         self.proj_layer = nn.Linear(self.in_features,self.out_features * self.head_num,bias = False)
#
#         # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
#         # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
#         self.score_fn_source = nn.Parameter(torch.Tensor(self.head_num,self.out_features,1))
#         self.score_fn_target = nn.Parameter(torch.Tensor(self.head_num,self.out_features,1))
#
#         # Bias is definitely not crucial to GAT - feel free to experiment
#         if args.bias and args.concat:
#             self.bias = nn.Parameter(torch.Tensor(self.head_num * self.out_features))
#         elif args.bias and not args.concat:
#             self.bias = nn.Parameter(torch.Tensor(self.out_features * self.head_num ))
#         else:
#             self.register_parameter('bias', None)
#
#         # # Use the module in three locations,before/after features projection and for attention coefficient
#         # self.dropout  = nn.Dropout(p = args.dropout)
#         # self.dropout2 = nn.Dropout(p = args.dropout)
#         # self.dropout3 = nn.Dropout(p=args.dropout)
#
#         self.LeakyReLU = nn.LeakyReLU(0.2)
#         self.activation = activation
#
#         self.rnn = GRUGate(self.head_num*self.out_features)
#
#         self.init_param()
#
#
#     def init_param(self):
#         """
#        The reason why using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
#            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
#        The original repo was developed in TensorFlow (TF) and they used the default initialization.
#        Feel free to experiment - there may be better initializations depending on your problem.
#        """
#         nn.init.xavier_normal_(self.proj_layer.weight)
#         nn.init.xavier_normal_(self.score_fn_source)
#         nn.init.xavier_normal_(self.score_fn_target)
#
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)
#
#     def forward(self,input,adj=None):
#         """
#         :param input: shape (b,agent_num,obs_num,obs_dim)
#         :param adj: shape (n,agent_num)
#         :return: features embedding
#         """
#         # Step 1: Linear projection and regularizer
#         bn,node_num,_ = input.shape
#
#         # assert adj.shape == (node_num,node_num), \
#         #     f'Expected connectivity matrix with shape=({node_num},{node_num}), got shape={adj.shape}.'
#
#         input = F.dropout(input,self.args.dropout,training=self.training)
#         # (b*agent_num,node_num,obs_dim) -> (b*agent_num,node_num,out_features,head_num)
#         proj_features = self.proj_layer(input).reshape(bn, node_num, self.out_features, self.head_num)
#
#         self.residual = proj_features.reshape(bn, node_num, self.out_features*self.head_num)
#
#         # (b * agent_num, node_num, out_features, head_num) -> (b*agent_num,head_num,node_num,out_features)
#         proj_features = proj_features.permute(0,3,1,2)
#
#         proj_features = F.dropout(proj_features,self.args.dropout,training=self.training)
#
#         # Step 2 Edg attention caluation
#         # (b*agent_num,head_num,node_num,out_features) * (1,head_num,out_features,1) ->
#         # (b*agent_num,head_num,node_num,1)
#         scores_source = torch.matmul(proj_features, self.score_fn_source.unsqueeze(0))
#         scores_target = torch.matmul(proj_features, self.score_fn_target.unsqueeze(0))
#
#         all_scores = self.LeakyReLU(scores_source + scores_target.transpose(3,2))
#
#         # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
#         # this will result in attention scores being computed only for existing edges
#         if adj is not None:
#             all_scores = all_scores + adj.unsqueeze(1)
#
#         # Step 3: Neighborhood aggregation
#         # shape (b*agent_num,head_num,node_num,node_num) * (b*agent_num,head_num,node_num,out_features)
#         all_attention_coefficients = F.softmax(all_scores,dim=-1)
#
#         out_node_features = torch.matmul(all_attention_coefficients,proj_features)
#
#         out_node_features = out_node_features.permute(0,2,1,3).contiguous().view(bn,node_num,self.head_num*self.out_features)
#
#         # Step 4: Residual/skip connections, concat and bias
#         out_nodes_features = self._skip_bias_cat(out_node_features)
#
#         return out_nodes_features
#
#
#     def _skip_bias_cat(self,out_nodes_features):
#         if not out_nodes_features.is_contiguous():
#             out_nodes_features = out_nodes_features.contiguous()
#
#         if self.skip_residual:
#             self.rnn(self.residual,out_nodes_features)
#
#         if self.bias is not None:
#             out_nodes_features += self.bias
#
#         return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
#
# class GAT_Critic(nn.Module):
#     def __init__(self,in_features,args):
#         super(GAT_Critic, self).__init__()
#         self.hidden_dim = args.hid_size
#         self.head_num = args.GAT_head_num
#         self.n_ = args.agent_num
#
#         self.out_features = args.hid_size // args.GAT_head_num
#
#         self.gat_layers = nn.ModuleList([GraphAttentionLayer(in_features,self.out_features,args),
#                                          GraphAttentionLayer(self.hidden_dim, self.out_features, args)
#                                          # GraphAttentionLayer(self.hidden_dim, self.out_features, args)
#                                          ])
#
#         self.reward_head = nn.Sequential(
#             nn.Linear(self.hidden_dim * self.n_, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, 1)
#         )
#         self.cost_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, 1)
#         )
#         if args.layernorm:
#             self.layernorm = nn.LayerNorm(self.hidden_dim)
#         if args.hid_activation == 'relu':
#             self.hid_activation = nn.ReLU()
#         elif args.hid_activation == 'tanh':
#             self.hid_activation = nn.Tanh()
#
#     def forward(self,input):
#         # shape (b*agent_num,node_num,out_features*head_num)
#
#         for layer in self.gat_layers:
#             input = layer(input)
#
#         pred_r = self.reward_head(
#             input.view(-1, self.n_ * self.hidden_dim))  # (b, 1)
#         pred_c = self.cost_head(input)  # (b,n,1)
#
#         return pred_r,pred_c
#





#
# import argparse
# import os.path as osp
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
# from torch_geometric.nn import GATConv
#
#
# class GAT_Critic(torch.nn.Module):
#     def __init__(self, in_channels,args):
#         super().__init__()
#
#         self.args = args
#         self.hidd_channels = args.hid_size
#         self.head_num = args.GAT_head_num
#         self.n_ = args.agent_num
#
#         self.conv1 = GATConv(in_channels, self.hidd_channels , self.head_num, dropout=0.6)
#         # On the Pubmed dataset, use `heads` output heads in `conv2`.
#         self.conv2 = GATConv(self.hidd_channels  * self.head_num, self.hidd_channels , heads=self.head_num,
#                              concat=True, dropout=0.6)
#
#         self.conv3 = GATConv(self.hidd_channels * self.head_num, self.hidd_channels, heads=self.head_num,
#                              concat=False, dropout=0.6)
#
#         self.reward_head = nn.Sequential(
#             nn.Linear(self.hidd_channels * self.n_, self.hidd_channels),
#             nn.ReLU(),
#             nn.Linear(self.hidd_channels, 1)
#         )
#         self.cost_head = nn.Sequential(
#             nn.Linear(self.hidd_channels, self.hidd_channels),
#             nn.ReLU(),
#             nn.Linear(self.hidd_channels, 1)
#         )
#         if args.layernorm:
#             self.layernorm = nn.LayerNorm(self.hidd_channels)
#         if args.hid_activation == 'relu':
#             self.hid_activation = nn.ReLU()
#         elif args.hid_activation == 'tanh':
#             self.hid_activation = nn.Tanh()
#
#     def forward(self, x,adj):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x,adj))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x,adj)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv3(x,adj).view(-1,self.n_,self.hidd_channels)
#
#         pred_r = self.reward_head(
#             x.view(-1, self.n_ * self.hidd_channels))  # (b, 1)
#         pred_c = self.cost_head(x)  # (b,n,1)
#         return pred_r,pred_c
#

from pickle import NONE
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = th.matmul(q, k.transpose(2, 3)) / np.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = th.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(EncoderLayer, self).__init__()

        self.n_heads = n_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        q = make_heads(self.Wq(x), self.n_heads)
        k = make_heads(self.Wk(x), self.n_heads)
        v = make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class TransformerCritic(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, input_shape, args):
        super(TransformerCritic, self).__init__()
        self.hidden_dim = args.hid_size
        # self.out_hidden_dim = args.out_hid_size
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.n_ = args.agent_num
        self.n_layers = args.n_layers
        self.attend_heads = args.attend_heads
        self.args = args
        self.init_projection_layer = nn.Linear(input_shape, self.hidden_dim)
        self.attn_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=self.hidden_dim,
                         n_heads=self.attend_heads)
            for _ in range(self.n_layers)
        ])
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.cost_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        if args.layernorm:
            self.layernorm = nn.LayerNorm(self.hidden_dim)
        if args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def forward(self, obs):
        # obs : (b, n, d)
        x = self.init_projection_layer(obs)  # x: (b,n,hidd_dim)
        for layer in self.attn_layers:
            x = layer(x)
        pred_r = self.reward_head(
            x.view(-1, self.n_ * self.hidden_dim))    # (b, 1)
        pred_c = self.cost_head(x)  # (b,n,1)
        return pred_r, pred_c




