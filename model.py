import torch.nn as nn
import torch
from z_encoder import Z_encoder
from h_encoder import H_encoder
from contrast import Contrast

class Model(nn.Module):
    def __init__(self, num_nodes, out_dim, K, tau, lam, feat_drop, attn_drop):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(num_nodes, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim)
        )
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x


        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.z = Z_encoder(num_nodes, out_dim, attn_drop)
        self.h = H_encoder(num_nodes, K, out_dim, attn_drop)
        self.contrast = Contrast(out_dim, tau, lam)

    def forward(self, adj, onehot, timestamp,t_span): 
        ini_feature = self.feat_drop(self.proj(onehot))
        z,a_z = self.z(adj,ini_feature,timestamp,t_span)
        h,a_h= self.h(adj,ini_feature)
        return z,h,a_z,a_h

    def loss(self, z, h, adj, neg):
        loss = self.contrast(z, h, adj, neg)
        return loss

    def get_next(self, adj, emb_z, emb_h, timestamp,a_z,train_snapshot,t_span):
        z = self.z.get_next_z(adj, emb_z, timestamp,a_z,train_snapshot,t_span)
        h,attn = self.h.get_next_h(emb_h)
        embeddings = torch.cat([z, h], dim=1)
        return embeddings,attn



