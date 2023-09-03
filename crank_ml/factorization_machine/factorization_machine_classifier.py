import torch
from torch import nn


class FactorizationMachineClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        n_classes=2,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        loss="log",
    ):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.loss = loss
        self.penalty_ = penalty
        self.alpha_ = alpha
        self.l1_ratio_ = l1_ratio
        self.n_classes = n_classes

        if n_classes == 2:
            self.output_target_shape = 1
            self.linear = nn.Linear(n_features, self.output_target_shape)
            self.embedding = nn.Linear(n_features, embed_dim)
            self.output_activation = nn.Sigmoid()
        elif n_classes > 2:
            self.output_target_shape = n_classes
            self.linear = nn.Linear(n_features, n_classes)
            self.embedding = nn.Linear(n_features, embed_dim * self.output_target_shape)
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

    def factorization_machine(self, embedding):
        square_of_sum = torch.sum(embedding, dim=1) ** 2
        sum_of_square = torch.sum(embedding**2, dim=1)
        output = square_of_sum - sum_of_square
        output = torch.sum(output, dim=1, keepdim=True)
        return output

    def forward(self, x):
        linear = self.linear(x)
        embedding = self.embedding(x).reshape(-1, self.embed_dim, self.output_target_shape)
        fm = self.factorization_machine(embedding)
        output = self.output_activation(linear + fm)
        return output
