import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


def normalize(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


class VMFTrainer:
    def __init__(self, dim, device="cpu"):
        self.mu_truth = torch.randn(dim).to(device)
        self.mu_hall = torch.randn(dim).to(device)
        self.device = device

    def set_initial_centroids(self, a, b):
        # a, b: torch tensors of shape [dim]
        self.mu_truth = F.normalize(a.to(self.device), dim=-1)
        self.mu_hall = F.normalize(b.to(self.device), dim=-1)

    def e_step(self, embeddings):
        emb = normalize(embeddings)
        t_score = (emb * self.mu_truth).sum(-1)
        h_score = (emb * self.mu_hall).sum(-1)
        probs = torch.softmax(torch.stack([t_score, h_score], dim=-1), dim=-1)
        return probs

    def m_step(self, embeddings, probs):
        emb = normalize(embeddings)
        self.mu_truth = normalize((probs[:, 0:1] * emb).sum(0))
        self.mu_hall = normalize((probs[:, 1:2] * emb).sum(0))


def train_tsv(hidden_states, labels=None, iters=10, init="random"):
    # Detect device from the incoming data
    device = hidden_states.device
    trainer = VMFTrainer(hidden_states.shape[-1], device=device)

    # optional supervised initialization for stability
    if init == "supervised" and labels is not None:
        labels_t = torch.as_tensor(labels, device=device)
        truth_mask = labels_t == 1
        hall_mask = labels_t == 0

        if truth_mask.any() and hall_mask.any():
            trainer.set_initial_centroids(
                hidden_states[truth_mask].mean(dim=0),
                hidden_states[hall_mask].mean(dim=0),
            )

    # optional k-means initialization for stability
    elif init == "kmeans":
        try:
            hs_cpu = hidden_states.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=2, random_state=0).fit(hs_cpu)
            centers = torch.from_numpy(kmeans.cluster_centers_).to(device)

            # choose assignment: cluster 0 -> truth, cluster 1 -> hall
            # this is arbitrary but provides a stable starting point
            trainer.set_initial_centroids(centers[0], centers[1])
        except Exception:
            # fallback to random if kmeans fails
            pass

    # if labels are available and we started supervised, use one labeled centroid fit
    if init == "supervised" and labels is not None:
        labels_t = torch.as_tensor(labels, device=device)
        probs = torch.zeros(hidden_states.shape[0], 2, device=device)
        probs[:, 0] = (labels_t == 1).float()
        probs[:, 1] = (labels_t == 0).float()
        trainer.m_step(hidden_states, probs)
        return trainer

    for _ in range(iters):
        probs = trainer.e_step(hidden_states)
        trainer.m_step(hidden_states, probs)

    return trainer
