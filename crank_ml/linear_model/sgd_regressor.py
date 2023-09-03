import torch
from torch import nn


class SGDRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
    ):
        super().__init__()
        self.n_features = n_features
        self.penalty_ = penalty
        self.alpha_ = alpha
        self.l1_ratio_ = l1_ratio
        self.weights = nn.Linear(n_features, 1)

    def penalty(self):
        if self.penalty_ == "l2":
            return sum([torch.sum(torch.linalg.norm(p)) for p in self.parameters()]) * self.alpha_
        elif self.penalty_ == "l1":
            return sum([torch.sum(torch.linalg.norm(p, p=1)) for p in self.parameters()]) * self.alpha_
        elif self.penalty_ == "elasticnet":
            l1_penalty = (self.l1_ratio_) * sum([torch.sum(torch.linalg.norm(p, p=1)) for p in self.parameters()])
            l2_penalty = (1 - self.l1_ratio_) * sum([torch.sum(torch.linalg.norm(p, p=2)) for p in self.parameters()])
            return (l1_penalty + l2_penalty) * self.alpha_
        return 0

    def forward(self, X):
        output = self.weights(X)
        return output
