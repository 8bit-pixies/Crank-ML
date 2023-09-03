from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class StandardScaler(nn.Module):
    def __init__(
        self,
        n_features: Optional[int] = None,
        tensor_mean: Optional[torch.Tensor] = None,
        tensor_var: Optional[torch.Tensor] = None,
        momentum=0.1,
        track_running_stats=False,
    ):
        super().__init__()
        self.n_features = n_features if n_features is not None else tensor_min.shape[0]  # inferred
        self.momentum = momentum
        tensor_mean = torch.zeros(self.n_features) if tensor_mean is None else tensor_mean
        tensor_var = torch.ones(self.n_features) if tensor_var is None else tensor_var

        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("tensor_mean", tensor_mean)
            self.register_buffer("tensor_var", tensor_var)
            # self.tensor_min: Optional[torch.Tensor]
            # self.tensor_min: Optional[torch.Tensor]
            self.tensor_mean = tensor_mean
            self.tensor_var = tensor_var
        else:
            self.tensor_mean = tensor_mean
            self.tensor_var = tensor_var

    def get_output_shape(self):
        return self.n_features

    def forward(self, X):
        if self.training and self.track_running_stats:
            self.tensor_mean = (1 - self.momentum) * self.tensor_mean + self.momentum * torch.mean(X, axis=0)
            self.tensor_var = (1 - self.momentum) * self.tensor_var + self.momentum * torch.var(X, axis=0)

        return (X - self.tensor_mean) / self.tensor_var
