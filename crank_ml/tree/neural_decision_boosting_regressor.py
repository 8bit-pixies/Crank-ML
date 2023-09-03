import random

import torch
from torch import nn


class NeuralDecisionTreeClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        depth=5,
        penalty="l2",
        used_features_rate=0.7,
        alpha=0.0001,
        gamma=0.1,
    ):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2**depth
        self.gamma = gamma
        self.penalty_ = penalty
        self.alpha_ = alpha

        num_used_features = int(n_features * used_features_rate)
        self.n_features = num_used_features
        one_hot = torch.eye(n_features)
        sampled_feature_indicies = sorted(random.sample(range(n_features), num_used_features))
        self.used_features_mask = one_hot[sampled_feature_indicies].T

        self.decision_layer = nn.Linear(self.n_features, self.n_leaves)
        self.decision_layer_bn = nn.BatchNorm1d(self.n_leaves)
        self.pi = torch.nn.Parameter(torch.rand(self.n_leaves, 1).float(), requires_grad=True)

    def _smoothstep(self, x):
        # using smoothstep instead of sigmoid
        s = (-2 / self.gamma**3) * (x**3) + (3 / (2 * self.gamma)) * (x) + 0.5
        s[x <= (-self.gamma / 2)] = 0
        s[x >= (self.gamma / 2)] = 1
        return s

    def forward(self, X):
        batch_size = X.size(0)
        X = torch.matmul(X, self.used_features_mask)  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = self._smoothstep(self.decision_layer_bn(self.decision_layer(X)))
        # Concatenate the routing probabilities with their complements.
        decisions = torch.stack([decisions, 1 - decisions], axis=2)  # [batch_size, num_leaves, 2]

        mu = torch.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = torch.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = torch.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[:, begin_idx:end_idx, :]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = torch.reshape(mu, [batch_size, self.n_leaves])  # [batch_size, num_leaves]
        probabilities = self.pi  # [num_leaves, num_classes]
        outputs = torch.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs


class NeuralDecisionForestRegressor(nn.Module):
    def __init__(
        self,
        n_estimators: int,
        n_features: int,
        depth=5,
        used_features_rate=0.7,
        penalty="l2",
        alpha=0.0001,
        gamma=0.1,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.penalty_ = penalty
        self.alpha_ = alpha
        self.trees = nn.ModuleList(
            [
                NeuralDecisionTreeClassifier(n_features, depth, penalty, used_features_rate, alpha, gamma)
                for _ in range(n_estimators)
            ]
        )
        # uses a randomly initialized batchnorm to estimate running weights.
        self.loss_batch_norm = nn.BatchNorm1d(1, momentum=None)

    def penalty(self):
        if self.penalty_ == "l2":
            return sum([torch.sum(torch.linalg.norm(p)) for p in self.parameters()]) * self.alpha_
        elif self.penalty_ == "l1":
            return sum([torch.sum(torch.linalg.norm(p, p=1)) for p in self.parameters()]) * self.alpha_
        return 0

    def train_forward(self, X, y):
        batch_size = X.size(0)
        self.loss_batch_norm.reset_parameters()
        weight = torch.ones([batch_size])

        for indx in range(self.n_estimators):
            y_hat = self.trees[indx].forward(X)

            loss_fn = nn.MSELoss(reduction="none")
            model_loss = loss_fn(y_hat, y) * weight

            if indx == 0:
                loss = torch.mean(model_loss, axis=0)
                # no gradient, for updating weights on initialization
                # as we're calculating running cumulative moving average
                self.loss_batch_norm(model_loss.unsqueeze(1)).detach()
            else:
                loss = loss + torch.mean(model_loss, axis=0)
            # change to -1, 1: 1 if instance is incorrect, -1 if it is correct
            sign_output = 1 - torch.tanh(self.loss_batch_norm(model_loss.unsqueeze(1))).squeeze(1)
            weight = weight * torch.pow(1 + self.learning_rate, sign_output)
        return torch.mean(loss)

    def forward(self, X):
        batch_size = X.size(0)
        outputs = torch.zeros([batch_size, 1])
        for indx in range(self.n_estimators):
            outputs += self.trees[indx].forward(X)
        outputs /= self.n_estimators
        return outputs
