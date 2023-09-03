import torch
from torch import nn


class SGDClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes=2,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        loss="log",
    ):
        super().__init__()
        self.n_features = n_features
        self.loss = loss
        self.penalty_ = penalty
        self.alpha_ = alpha
        self.l1_ratio_ = l1_ratio
        self.n_classes = n_classes

        if n_classes == 2:
            self.weights = nn.Linear(n_features, 1)
            self.output_activation = nn.Sigmoid()
        elif n_classes > 2:
            self.weights = nn.Linear(n_features, n_classes)
            self.output_activation = nn.Softmax()
        else:
            raise ValueError("n_classes must be 2 or more. Got:", n_classes)

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
        output = self.output_activation(self.weights(X))
        return output
