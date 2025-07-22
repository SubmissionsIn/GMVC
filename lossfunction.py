import torch
import torch.nn as nn
import math


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")



    def mask_correlated_samples(self, N, A):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        half_N = N // 2
        indices = torch.arange(half_N)
        mask[indices, half_N + indices] = 0
        mask[half_N + indices, indices] = 0
        row_indices, col_indices = torch.where(A == 1)
        mask[row_indices, half_N + col_indices] = 0
        mask[row_indices, col_indices] = 0
        mask[half_N + row_indices, col_indices] = 0
        mask[half_N + row_indices, half_N + row_indices] = 0
        mask = mask.bool()
        return mask

    def mask_correlated_samples_label(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        half_N = N // 2
        indices = torch.arange(half_N)
        mask[indices, half_N + indices] = 0
        mask[half_N + indices, indices] = 0
        mask = mask.bool()
        return mask


    def feature_contrastive_loss(self, h_v, h_w, A, gamma):
        N = 2 * self.batch_size
        h = torch.cat((h_v, h_w), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f

        S_pos_inter_i_j = torch.diag(sim, self.batch_size)
        S_pos_inter_j_i = torch.diag(sim, -self.batch_size)

        S_pos_inter = torch.cat((S_pos_inter_i_j, S_pos_inter_j_i), dim=0).reshape(N, 1)
        S_pos_intra_i_i = torch.sum(sim[:self.batch_size, :self.batch_size] * A, dim=1)
        S_pos_intra_i_j = torch.sum(sim[:self.batch_size, self.batch_size:] * A, dim=1)
        S_pos_intra_i = S_pos_intra_i_i + S_pos_intra_i_j
        S_pos_intra_j_i = torch.sum(sim[self.batch_size:, :self.batch_size] * A, dim=1)
        S_pos_intra_j_j = torch.sum(sim[self.batch_size:, self.batch_size:] * A, dim=1)
        S_pos_intra_j = S_pos_intra_j_i + S_pos_intra_j_j
        S_pos_intra = torch.cat((S_pos_intra_i, S_pos_intra_j), dim=0).reshape(N, 1)
        positive_samples = S_pos_inter + gamma * S_pos_intra
        mask = self.mask_correlated_samples(N, A)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


    def forward_label(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()

        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()

        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples_label(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy


    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)

        return (weight.t() / torch.sum(weight, 1)).t()
