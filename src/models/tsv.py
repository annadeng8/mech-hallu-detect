import torch
import torch.nn as nn
import torch.nn.functional as F


class TSVProbe(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.tsv = nn.Parameter(torch.zeros(hidden_size))
        self.truth_centroid = nn.Parameter(torch.randn(hidden_size))
        self.halluc_centroid = nn.Parameter(torch.randn(hidden_size))

    def transform(self, h):
        return F.normalize(h + self.tsv, dim=-1)

    def score(self, h):
        z = self.transform(h)
        truth = F.normalize(self.truth_centroid, dim=-1)
        hall = F.normalize(self.halluc_centroid, dim=-1)
        truth_sim = (z * truth).sum(-1)
        hall_sim = (z * hall).sum(-1)
        return truth_sim - hall_sim

    def conflict_score(self, h):
        return torch.sigmoid(-self.score(h))

    def forward(self, h):
        return self.score(h)
