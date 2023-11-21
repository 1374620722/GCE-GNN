import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.num_heads = 3

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim) #a_input = [batch_size, N, N, self.dim]

        e_0 = torch.matmul(a_input, self.a_0) #e_0 = [batch_size, N, N, 1]
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0) #mask = [batch_size, N, N]
        alpha = torch.where(adj.eq(1), e_0, mask) #adj = [batch_size, N, N]
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1) #alpha =  [batch_size, N, N]

        output = torch.matmul(alpha, h) #h = [batch_size, N,dim]
        return output # output = [batch_size, N,dim]

# class LocalAggregator(nn.Module):
#     def __init__(self, dim, alpha, dropout=0., name=None):
#         super(LocalAggregator, self).__init__()
#         self.dim = dim
#         self.dropout = dropout
#         self.num_heads = 4
#         self.head_dim = dim // 4
#
#         self.a = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim, 4))
#         self.bias = nn.Parameter(torch.Tensor(self.dim))
#
#         self.leakyrelu = nn.LeakyReLU(alpha)
#
#     def forward(self, hidden, adj, mask_item=None):
#         batch_size = hidden.shape[0]
#         N = hidden.shape[1]
#
#         h = hidden.view(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)
#         # h = [batch_size, num_heads, N, head_dim]
#
#         a_input = (h.repeat(1, 1, N, 1).view(batch_size, self.num_heads, N * N, self.head_dim)
#                    * h.repeat(1, 1, N, 1)).view(batch_size, self.num_heads, N, N, self.head_dim)
#         # a_input = [batch_size, num_heads, N, N, head_dim]
#
#         e = torch.matmul(a_input, self.a)
#         # e = [batch_size, num_heads, N, N, 4]
#
#         e = self.leakyrelu(e).sum(dim=-1).view(batch_size, self.num_heads, N, N)
#         # e = [batch_size, num_heads, N, N]
#
#         mask = -9e15 * torch.ones_like(e)
#         alpha = torch.where(adj.eq(1), e, mask)
#         alpha = torch.where(adj.eq(2), e, alpha)
#         alpha = torch.where(adj.eq(3), e, alpha)
#         alpha = torch.where(adj.eq(4), e, alpha)
#         alpha = torch.softmax(alpha, dim=-1)
#         # alpha = [batch_size, num_heads, N, N]
#
#         output = torch.matmul(alpha, h.transpose(1, 2)).transpose(1, 2).reshape(batch_size, N, self.dim)
#         # output = [batch_size, N, dim]
#
#         return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output