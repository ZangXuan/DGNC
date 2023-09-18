import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable

class TimeEncode(nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()

        self.expand_dim = expand_dim
        self.time_dim = 2 * expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.expand_dim))).float())


    def forward(self, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]

        time_emb = torch.zeros(batch_size, seq_len, self.time_dim).cuda()  # [N, L, 1]

        time_emb[:, :, 0::2] = torch.sin(map_ts)
        time_emb[:, :, 1::2] = torch.cos(map_ts)


        return time_emb

class GraphAttentionLayer_edit(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes, dropout, alpha=0.05, concat=True):
        super(GraphAttentionLayer_edit, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.layer_norm = nn.LayerNorm(self.out_dim, eps=1e-6)

        self.t_encoder = TimeEncode(self.out_dim)

    def forward(self, ini_feature, adj, timestamp, a, time_end):
        z = torch.mm(ini_feature, self.W) # shape [N, out_features]

        N = z.size()[0]

        a_input = torch.cat([z.repeat(1, N).view(N * N, -1), z.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_dim) # shape[N, N, 2*out_features]

        time_emb = self.t_encoder(time_end * torch.ones_like(timestamp).cuda() - timestamp)
        e = self.leakyrelu(torch.matmul(a_input+time_emb, a).squeeze(2))  # [N,N,1] -> [N,N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        temp = torch.matmul(attention, z)  # [N,N], [N, out_features] --> [N, out_features]

        temp += z
        z_emb = self.layer_norm(temp)

        return z_emb



class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        lb, ub = -math.sqrt(1/hidden_size), math.sqrt(1/hidden_size) 
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(3)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])

    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low)  # 按照官方的初始化方法来初始化网络参数
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

    def forward(self, x, hid):
        r = torch.sigmoid(torch.mm(x, self.in2hid_w[0]) + self.in2hid_b[0] +
                          torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        z = torch.sigmoid(torch.mm(x, self.in2hid_w[1]) + self.in2hid_b[1] +
                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])
        n = torch.tanh(torch.mm(x, self.in2hid_w[2]) + self.in2hid_b[2] +
                       torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))
        next_hid = torch.mul((1 - z), n) + torch.mul(z, hid)
        return next_hid

class Z_encoder(nn.Module):
    def __init__(self, num_nodes, out_dim, attn_drop):
        super(Z_encoder, self).__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim

        self.agg = GraphAttentionLayer_edit(out_dim, out_dim, num_nodes, attn_drop)
        self.rnn_a = GRUCell(out_dim,out_dim*2)


    def forward(self, adj, feature,timestamp,t_span):
        num_snapshot = len(adj)
        z_emb = [Variable(torch.FloatTensor(self.num_nodes, self.out_dim)) for _ in range(num_snapshot)]
        a = Variable(torch.randn(size=(2 * self.out_dim, 1)).cuda(), requires_grad=True)

        for i in range(num_snapshot):
            time_end = (i + 1) * t_span
            if i == 0:
                z_emb[i] = self.agg(feature, adj[i], timestamp[i],a,time_end)

            else:
                z_emb[i] = self.agg(z_emb[i - 1], adj[i], timestamp[i],a,time_end)


            a_hid = a.t().repeat(self.num_nodes, 1)
            a_next = self.rnn_a(z_emb[i], a_hid)
            a = torch.mean(a_next, 0, True).t()

        return z_emb, a

    def get_next_z(self, adj, emb, timestamp,a,train_snapshot,t_span):
        time_end = (train_snapshot + 1) * t_span
        z_emb_next = self.agg(emb, adj, timestamp, a, time_end)
        return z_emb_next



