import torch
import torch.nn as nn

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        # self.proj = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    # def forward(self, z, h, pos, neg):
    #     num_snap = len(z)
    #     loss_z = torch.zeros(num_snap)
    #     loss_h = torch.zeros(num_snap)
    #     loss_zz = torch.zeros(num_snap)
    #     loss_hh = torch.zeros(num_snap)
    #     for i in range(num_snap):
    #         z_proj = self.proj(z[i])
    #         h_proj = self.proj(h[i])
    #         matrix_z2h = self.sim(z_proj, h_proj)
    #         matrix_h2z = matrix_z2h.t()
    #
    #         matrix_z2z = self.sim(z_proj, z_proj)
    #         matrix_h2h = self.sim(h_proj, h_proj)
    #
    #         # matrix_z2h = matrix_z2h / (torch.sum(matrix_z2h, dim=1).view(-1, 1) + 1e-8)
    #         # lori_z = -torch.log(matrix_z2h.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
    #         #
    #         # matrix_h2z = matrix_h2z / (torch.sum(matrix_h2z, dim=1).view(-1, 1) + 1e-8)
    #         # lori_h = -torch.log(matrix_h2z.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
    #
    #         matrix_z2h = matrix_z2h / (torch.sum(matrix_z2h.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
    #         lori_z = -torch.log(matrix_z2h.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
    #
    #         matrix_h2z = matrix_h2z / (torch.sum(matrix_h2z.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
    #         lori_h = -torch.log(matrix_h2z.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
    #
    #         matrix_z2z = matrix_z2z / (torch.sum(matrix_z2z.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
    #         lori_zz = -torch.log(matrix_z2z.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
    #
    #         matrix_h2h = matrix_h2h / (torch.sum(matrix_h2h.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
    #         lori_hh = -torch.log(matrix_h2h.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
    #
    #         loss_z[i] = lori_z
    #         loss_h[i] = lori_h
    #         loss_zz[i] = lori_zz
    #         loss_hh[i] = lori_hh
    #
    #     loss_z = loss_z.mean()
    #     loss_h = loss_h.mean()
    #     loss_zz = loss_zz.mean()
    #     loss_hh = loss_hh.mean()
    #
    #     # loss = self.lam * loss_z + (1 - self.lam) * loss_h
    #     loss = self.lam * (loss_z+loss_h) + (1 - self.lam) * (loss_zz+loss_hh)
    #
    #     return loss

    def forward(self, z, h, pos, neg):
        num_snap = len(z)
        # loss_z = torch.zeros(num_snap)
        # loss_h = torch.zeros(num_snap)
        # loss_zz = torch.zeros(num_snap)
        # loss_hh = torch.zeros(num_snap)
        loss = torch.zeros(num_snap)
        for i in range(num_snap):
            z_proj = self.proj(z[i])
            h_proj = self.proj(h[i])
            matrix_z2h = self.sim(z_proj, h_proj)
            matrix_h2z = matrix_z2h.t()

            matrix_z2z = self.sim(z_proj, z_proj)
            matrix_h2h = self.sim(h_proj, h_proj)

            matrix_sim = self.lam * (matrix_z2h+matrix_h2z) + (1 - self.lam) * (matrix_z2z +matrix_h2h)

            # matrix_z2h = matrix_z2h / (torch.sum(matrix_z2h, dim=1).view(-1, 1) + 1e-8)
            # lori_z = -torch.log(matrix_z2h.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
            #
            # matrix_h2z = matrix_h2z / (torch.sum(matrix_h2z, dim=1).view(-1, 1) + 1e-8)
            # lori_h = -torch.log(matrix_h2z.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()

            matrix_sim = matrix_sim / (torch.sum(matrix_sim.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
            lori = -torch.log(matrix_sim.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()

            # matrix_z2h = matrix_z2h / (torch.sum(matrix_z2h.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
            # lori_z = -torch.log(matrix_z2h.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
            #
            # matrix_h2z = matrix_h2z / (torch.sum(matrix_h2z.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
            # lori_h = -torch.log(matrix_h2z.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
            #
            # matrix_z2z = matrix_z2z / (torch.sum(matrix_z2z.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
            # lori_zz = -torch.log(matrix_z2z.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()
            #
            # matrix_h2h = matrix_h2h / (torch.sum(matrix_h2h.mul(neg[i] + pos[i].to_dense()), dim=1).view(-1, 1) + 1e-8)
            # lori_hh = -torch.log(matrix_h2h.mul(pos[i].to_dense()).sum(dim=-1) + 1e-6).mean()

            # loss_z[i] = lori_z
            # loss_h[i] = lori_h
            # loss_zz[i] = lori_zz
            # loss_hh[i] = lori_hh

            loss[i] = lori

        # loss_z = loss_z.mean()
        # loss_h = loss_h.mean()
        # loss_zz = loss_zz.mean()
        # loss_hh = loss_hh.mean()
        loss = loss.mean()

        # loss = self.lam * loss_z + (1 - self.lam) * loss_h
        # loss = self.lam * (loss_z+loss_h) + (1 - self.lam) * (loss_zz+loss_hh)

        return loss
