import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable

class Community(nn.Module):
    def __init__(self, num_nodes, K, out_dim):
        super(Community, self).__init__()

        self.K = K
        self.out_dim = out_dim

        self.W_ini = nn.Parameter(torch.zeros(size=(out_dim, out_dim)))
        nn.init.xavier_normal_(self.W_ini.data, gain=1)
        self.mean = nn.Parameter(torch.zeros(size=(K, num_nodes)))
        nn.init.xavier_normal_(self.mean.data, gain=1)
        self.log_var = nn.Parameter(torch.zeros(size=(K, num_nodes)))
        nn.init.xavier_normal_(self.log_var.data, gain=1)
        self.prelu = nn.PReLU()

    def forward(self, feature):
        temp = torch.tanh(torch.mm(feature, self.W_ini))
        mean = torch.tanh(torch.mm(self.mean,temp))
        log_vars = torch.tanh(torch.mm(self.log_var,temp))
        community = torch.tanh(mean + torch.randn(self.K, self.out_dim).cuda() * torch.exp(log_vars / 2))

        return community



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class H_Attention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model//n_head
        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.fc = nn.Linear(n_head * self.d_k, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, n_head = self.d_k, self.n_head
        len_q, len_k, len_v = q.size(0), k.size(0), v.size(0)
        residual = q
        q = self.w_qs(q).view(len_q, n_head, d_k)
        k = self.w_ks(k).view(len_k, n_head, d_k)
        v = self.w_vs(v).view(len_v, n_head, d_k)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(0, 1).contiguous().view(len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class H_encoder(nn.Module):
    def __init__(self, num_nodes, K, out_dim,attn_drop):
        super(H_encoder, self).__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.K = K

        self.Community = Community(num_nodes, K, out_dim)
        self.H_Attention = H_Attention(4, out_dim, attn_drop)
        self.prelu = nn.PReLU()



    def forward(self, adj, feature):
        num_snapshot = len(adj)
        attn_list = []
        h_emb = [Variable(torch.FloatTensor(self.num_nodes, self.out_dim)) for _ in range(num_snapshot)]
       
        for i in range(num_snapshot):
            if i == 0:
                community = self.Community(feature)
                h_emb[i],attn = self.H_Attention(feature, community, community)
            else:
                community = self.Community(h_emb[i - 1])
                h_emb[i],attn  = self.H_Attention(feature, community, community)
            attn_list.append(attn)

        return h_emb,attn_list

    def get_next_h(self, emb):
        community = self.Community(emb)
        h_emb_next,attn = self.H_Attention(emb, community,  community)

        return h_emb_next,attn









