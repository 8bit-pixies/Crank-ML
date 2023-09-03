from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class KBinsDiscretizer(nn.Module):
    def __init__(
        self,
        n_bins: int = 255,
        n_features: Optional[int] = None,
        tensor_min: Optional[torch.Tensor] = None,
        tensor_max: Optional[torch.Tensor] = None,
        momentum=0.1,
        encode="ordinal",
        track_running_stats=False,
    ):
        """Pytorch Module which binarizers continuous variables. The output is ordinal.

        This is to ensure the input and output dimensions stay the same; one-hot output can be constructed through:
        `F.one_hot(x, num_classes=self.n_bins)`

        Args:
            n_bins (int, optional): Number of bins to binarizer. Defaults to 255.
            tensor_min (torch.Tensor, optional): Tensor containing the minimum range. Defaults to None.
            tensor_max (torch.Tensor, optional): Tensor containing the maximum range. Defaults to None.
            momentum (float, optional): Momentum when updating running stats.
                See pytorch docs on BatchNorm. Defaults to 0.1.
            track_running_stats (bool, optional): Whether to keep stats running or keep it static.
                See pytorch docs on BatchNorm. Defaults to False.
        """
        super().__init__()
        self.n_features = n_features if n_features is not None else tensor_min.shape[0]  # inferred
        tensor_min = torch.zeros(self.n_features) if tensor_min is None else tensor_min
        tensor_max = torch.ones(self.n_features) if tensor_max is None else tensor_max

        self.n_bins = n_bins
        self.momentum = momentum
        if encode not in {"onehot", "ordinal", "onehot-flatten"}:
            raise ValueError(f"Parameter encode should be one in ['onehot', 'ordinal']. Got: {encode}")
        self.encode = encode
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("tensor_min", tensor_min)
            self.register_buffer("tensor_max", tensor_max)
            # self.tensor_min: Optional[torch.Tensor]
            # self.tensor_min: Optional[torch.Tensor]
            self.tensor_min = tensor_min
            self.tensor_max = tensor_max
        else:
            self.tensor_min = tensor_min
            self.tensor_max = tensor_max

    def get_output_shape(self):
        if self.encode == "ordinal":
            return self.n_features
        if self.encode == "onehot":
            return (self.n_features, self.n_bins)
        if self.encode == "onehot-flatten":
            return self.n_features * self.n_bins

    def forward(self, X):
        if self.training and self.track_running_stats:
            self.tensor_min = (1 - self.momentum) * self.tensor_min + self.momentum * torch.min(X, axis=0).values
            self.tensor_max = (1 - self.momentum) * self.tensor_max + self.momentum * torch.max(X, axis=0).values

        X_std = ((X - self.tensor_min) / (self.tensor_max - self.tensor_min)) * self.n_bins
        output = torch.clamp(X_std.long(), min=0, max=self.n_bins - 1)
        if self.encode == "ordinal":
            return output
        elif self.encode == "onehot":
            return F.one_hot(output, num_classes=self.n_bins)
        elif self.encode == "onehot-flatten":
            return F.one_hot(output, num_classes=self.n_bins).reshape(-1, self.n_features * self.n_bins)
