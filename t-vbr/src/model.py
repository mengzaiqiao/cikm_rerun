import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(12345)


class VAE_D(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        time_step,
        user_fea,
        item_fea,
        late_dim,
        emb_dim,
        neg_n,
        batch_size,
        activator="tanh",
        alpha=0.0,
        device=torch.device("cpu"),
    ):
        super(VAE_D, self).__init__()
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.time_step = time_step
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.late_dim = late_dim
        self.emb_dim = emb_dim
        self.neg_n = neg_n
        self.batch_size = batch_size
        self.esp = 1e-10
        print("self.device in VAE_D", self.device)

        if activator == "tanh":
            self.act = nn.Tanh
        elif activator == "sigmoid":
            self.act = nn.Sigmoid
        elif activator == "relu":
            self.act = nn.Relu
        else:
            self.act = nn.Tanh
        self.alpha = alpha
        self.user_fea_dim = self.user_fea.shape[1]
        self.item_fea_dim = self.item_fea.shape[1]
        print("input feature shape", self.user_fea_dim, self.item_fea_dim)
        self.init_layers()

    def init_layers(self):
        self.time_embdding = nn.Embedding(
            self.time_step + 1,
            self.time_step + 1,
            _weight=torch.eye(self.time_step + 1, self.time_step + 1),
        )

        self.user_mean = nn.Embedding(
            self.n_users, self.emb_dim, _weight=torch.ones(self.n_users, self.emb_dim)
        )
        self.user_std = nn.Embedding(
            self.n_users, self.emb_dim, _weight=torch.zeros(self.n_users, self.emb_dim)
        )

        self.item_mean = nn.Embedding(
            self.n_items, self.emb_dim, _weight=torch.ones(self.n_items, self.emb_dim)
        )
        self.item_std = nn.Embedding(
            self.n_items, self.emb_dim, _weight=torch.zeros(self.n_items, self.emb_dim)
        )

        self.time2mean_u = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.user_fea_dim, self.emb_dim
            ),
            self.act(),
        )
        self.time2std_u = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.user_fea_dim, self.emb_dim
            ),
            self.act(),
        )
        self.time2mean_i = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.item_fea_dim, self.emb_dim
            ),
            self.act(),
        )
        self.time2std_i = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.item_fea_dim, self.emb_dim
            ),
            self.act(),
        )

    def user_encode(self, index, time_laten, pri_time_laten):
        user_mean = self.user_mean(index).squeeze(1)  # (batch, out_size)
        user_mean_pri = self.time2mean_u(
            torch.cat([user_mean, pri_time_laten, self.user_fea[index]], 1)
        )
        user_mean = self.time2mean_u(
            torch.cat([user_mean, time_laten, self.user_fea[index]], 1)
        )

        user_std = self.user_std(index).squeeze(1)
        user_std_pri = (
            self.time2std_u(
                torch.cat([user_std, pri_time_laten, self.user_fea[index]], 1)
            )
            .mul(0.5)
            .exp()
        )
        user_std = (
            self.time2std_u(torch.cat([user_std, time_laten, self.user_fea[index]], 1))
            .mul(0.5)
            .exp()
        )
        return ((user_mean_pri, user_std_pri), (user_mean, user_std))

    def item_encode(self, index, time_laten, pri_time_laten):
        item_mean = self.item_mean(index).squeeze(1)  # (batch, out_size)
        #         print(item_mean)
        item_mean_pri = self.time2mean_i(
            torch.cat([item_mean, pri_time_laten, self.item_fea[index]], 1)
        )
        item_mean = self.time2mean_i(
            torch.cat([item_mean, time_laten, self.item_fea[index]], 1)
        )
        #         print(item_mean)

        item_std = self.item_std(index).squeeze(1)  # (batch, out_size)
        item_std_pri = (
            self.time2std_i(
                torch.cat([item_std, pri_time_laten, self.item_fea[index]], 1)
            )
            .mul(0.5)
            .exp()
        )
        item_std = (
            self.time2std_i(torch.cat([item_std, time_laten, self.item_fea[index]], 1))
            .mul(0.5)
            .exp()
        )
        return ((item_mean_pri, item_std_pri), (item_mean, item_std))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def DKL(self, dis1, dis2, neg=False):
        mean1, std1 = dis1
        mean2, std2 = dis2
        var1 = std1.pow(2) + self.esp
        var2 = std2.pow(2) + self.esp
        mean_pow2 = (
            (mean2 - mean1)
            * (torch.tensor(1.0, device=self.device) / var2)
            * (mean2 - mean1)
        )
        tr_std_mul = (torch.tensor(1.0, device=self.device) / var2) * var1
        if neg == False:
            dkl = (
                (torch.log(var2 / var1) - 1 + tr_std_mul + mean_pow2)
                .mul(0.5)
                .sum(dim=1)
                .mean()
            )
        else:
            dkl = (
                (torch.log(var2 / var1) - 1 + tr_std_mul + mean_pow2)
                .mul(0.5)
                .sum(dim=2)
                .sum(dim=1)
                .mean()
            )
        return dkl

    def forward(
        self, pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2, pos_batch_t, neg_batch_t
    ):

        """
        time embedding
        """
        pos_time_laten = self.time_embdding(
            pos_batch_t + torch.tensor(1).to(self.device)
        ).squeeze(1)
        pos_pri_time_laten = self.time_embdding(pos_batch_t)

        neg_time_laten = self.time_embdding(
            neg_batch_t + torch.tensor(1).to(self.device)
        ).squeeze(1)
        neg_pri_time_laten = self.time_embdding(neg_batch_t)

        """
        positive user embeddings
        """

        pos_u_dis_pri, pos_u_dis = self.user_encode(
            pos_u, pos_time_laten, pos_pri_time_laten
        )
        pos_u_emb = self.reparameterize(pos_u_dis[0], pos_u_dis[1])
        pos_u_kl = self.DKL(pos_u_dis_pri, pos_u_dis, False)

        """
        positive item embeddings
        """
        pos_i_1_dis_pri, pos_i_1_dis = self.item_encode(
            pos_i_1, pos_time_laten, pos_pri_time_laten
        )
        pos_i_1_emb = self.reparameterize(pos_i_1_dis[0], pos_i_1_dis[1])
        pos_i_1_kl = self.DKL(pos_i_1_dis_pri, pos_i_1_dis, False)

        pos_i_2_dis_pri, pos_i_2_dis = self.item_encode(
            pos_i_2, pos_time_laten, pos_pri_time_laten
        )
        pos_i_2_emb = self.reparameterize(pos_i_2_dis[0], pos_i_2_dis[1])
        pos_i_2_kl = self.DKL(pos_i_2_dis_pri, pos_i_2_dis, False)

        """
        negative user embeddings
        """
        neg_u_dis_pri, neg_u_dis = self.user_encode(
            neg_u.view(-1), neg_time_laten, neg_pri_time_laten
        )
        neg_u_emb = self.reparameterize(neg_u_dis[0], neg_u_dis[1]).view(
            -1, self.neg_n, self.emb_dim
        )
        neg_u_kl = self.DKL(neg_u_dis_pri, neg_u_dis, False)

        """
        negative item embeddings
        """
        neg_i_1_dis_pri, neg_i_1_dis = self.item_encode(
            neg_i_1.view(-1), neg_time_laten, neg_pri_time_laten
        )
        neg_i_1_emb = self.reparameterize(neg_i_1_dis[0], neg_i_1_dis[1]).view(
            -1, self.neg_n, self.emb_dim
        )
        neg_i_1_kl = self.DKL(neg_i_1_dis_pri, neg_i_1_dis, False)

        neg_i_2_dis_pri, neg_i_2_dis = self.item_encode(
            neg_i_2.view(-1), neg_time_laten, neg_pri_time_laten
        )
        neg_i_2_emb = self.reparameterize(neg_i_2_dis[0], neg_i_2_dis[1]).view(
            -1, self.neg_n, self.emb_dim
        )
        neg_i_2_kl = self.DKL(neg_i_2_dis_pri, neg_i_2_dis, False)

        input_emb_u = pos_i_1_emb + pos_i_2_emb

        u_pos_score = torch.mul(pos_u_emb, input_emb_u).squeeze()
        u_pos_score = torch.sum(u_pos_score, dim=1)
        u_pos_score = F.logsigmoid(u_pos_score)

        u_neg_score = torch.bmm(neg_u_emb, pos_u_emb.unsqueeze(2)).squeeze()
        u_neg_score = F.logsigmoid(-1 * u_neg_score)
        u_score = -1 * (torch.sum(u_pos_score) + torch.sum(u_neg_score))

        input_emb_i_1 = pos_u_emb + pos_i_2_emb
        i_1_pos_score = torch.mul(pos_i_1_emb, input_emb_i_1).squeeze()
        i_1_pos_score = torch.sum(i_1_pos_score, dim=1)
        i_1_pos_score = F.logsigmoid(i_1_pos_score)
        i_1_neg_score = torch.bmm(neg_i_1_emb, pos_i_1_emb.unsqueeze(2)).squeeze()
        i_1_neg_score = F.logsigmoid(-1 * i_1_neg_score)
        i_1_score = -1 * (torch.sum(i_1_pos_score) + torch.sum(i_1_neg_score))

        input_emb_i_2 = pos_u_emb + pos_i_1_emb
        i_2_pos_score = torch.mul(pos_i_2_emb, input_emb_i_2).squeeze()
        i_2_pos_score = torch.sum(i_2_pos_score, dim=1)
        i_2_pos_score = F.logsigmoid(i_2_pos_score)

        i_2_neg_score = torch.bmm(neg_i_2_emb, pos_i_2_emb.unsqueeze(2)).squeeze()
        i_2_neg_score = F.logsigmoid(-1 * i_2_neg_score)
        i_2_score = -1 * (torch.sum(i_2_pos_score) + torch.sum(i_2_neg_score))

        self.reconstruct = (u_score + i_1_score + i_2_score) / (self.batch_size)
        self.kl_loss = -0.5 * (
            pos_u_kl + pos_i_1_kl + pos_i_2_kl + neg_u_kl + neg_i_1_kl + neg_i_2_kl
        )

        return (1 - self.alpha) * self.reconstruct + (self.alpha * self.kl_loss)

    def predict(self, users, items, t=0):
        with torch.no_grad():
            users = torch.tensor(users, dtype=torch.int64, device=self.device)

            items = torch.tensor(items, dtype=torch.int64, device=self.device)

            times = torch.tensor(
                [self.time_step] * len(users), dtype=torch.int64, device=self.device
            )
            """
            time embedding
            """
            time_laten = self.time_embdding(times).squeeze(1)
            pri_time_laten = self.time_embdding(times - 1)

            """
            positive user embeddings
            """
            pos_u_dis_pri, pos_u_dis = self.user_encode(
                users, time_laten, pri_time_laten
            )

            """
            positive item embeddings
            """
            pos_i_dis_pri, pos_i_dis = self.item_encode(
                items, time_laten, pri_time_laten
            )

            scores = torch.bmm(
                pos_u_dis[0].view(len(users), 1, self.emb_dim),
                pos_i_dis[0].view(len(users), self.emb_dim, 1),
            ).squeeze()
            #             print(scores)
            if self.device.type == "cuda":
                return scores.cpu().detach().numpy()
            else:
                return scores.detach().numpy()
