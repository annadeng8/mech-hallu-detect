import torch


def sinkhorn(logits, n_iters=10, epsilon=0.1):
    Q = torch.exp(logits / epsilon)
    Q /= Q.sum()

    for _ in range(n_iters):
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= Q.sum(dim=1, keepdim=True)

    return Q