import torch
from torch import nn


class FieldawareFactorizationMachineClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        n_latent_factors: int = 10,
        n_classes=2,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
    ):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.n_latent_factors = n_latent_factors
        self.penalty_ = penalty
        self.alpha_ = alpha
        self.l1_ratio_ = l1_ratio
        self.n_classes = n_classes

        if n_classes == 2:
            self.output_target_shape = 1
            self.linear = nn.Linear(n_features, self.output_target_shape)
            self.embeddings = nn.ModuleList([nn.Linear(n_features, embed_dim) for _ in range(self.n_latent_factors)])
            self.output_activation = nn.Sigmoid()
        elif n_classes > 2:
            self.output_target_shape = n_classes
            self.linear = nn.Linear(n_features, n_classes)
            self.embeddings = nn.ModuleList(
                [nn.Linear(n_features, embed_dim * self.output_target_shape) for _ in range(self.n_latent_factors)]
            )
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

    def field_factorization_machine(self, x):
        xs = [
            self.embeddings[i](x).reshape(-1, self.embed_dim, self.output_target_shape)
            for i in range(self.n_latent_factors)
        ]
        ix = list()
        for i in range(self.n_latent_factors - 1):
            for j in range(i + 1, self.n_latent_factors):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        ffm_term = torch.sum(ix, dim=1)
        return ffm_term

    def forward(self, x):
        linear = self.linear(x)
        ffm = self.field_factorization_machine(x)
        output = self.output_activation(linear + ffm)
        return output
