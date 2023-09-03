import torch
from torch import nn


class PCA(nn.Module):
    """
    PCA via polyak averaging the inputs. It may not be stable
    """

    def __init__(self, n_features: int, n_components=6, polyak_weight=0.1):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.polyak_weight = polyak_weight
        self.V = torch.nn.Parameter(torch.zeros((self.n_features, self.n_components)).float(), requires_grad=False)

    def forward(self, X):
        if self.training:
            _, _, V = torch.pca_lowrank(X, q=self.n_components, center=False)
            if torch.abs(torch.sum(self.V)) == 0:
                self.V = nn.Parameter(V)
            else:
                self.V = nn.Parameter((1 - self.polyak_weight) * self.V + self.polyak_weight * V)
        return torch.matmul(X, self.V)
