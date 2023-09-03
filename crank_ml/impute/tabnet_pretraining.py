"""
TabNet pretraining example. This should be combined with the classifier (req. example).

Workflow is:
1. run pretraining
2. train by first transferring weights and imputing where necessary
"""

import math

import torch
from torch import nn


class GhostBatchNorm(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super().__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(math.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        non_zeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * non_zeros, dim=dim) / torch.sum(non_zeros, dim=dim)
        self.grad_input = non_zeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class AttentiveTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        group_dim,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()
        self.fc = nn.Linear(input_dim, group_dim, bias=False)
        self.bn = GhostBatchNorm(group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.sparsemax(x)
        return x


class FeatureTransformer(nn.Module):
    """
    Feature Transformer is an Independent GLU block, specific to each step
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_glu=2,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()

        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        self.glu_layers.append(GatedLinearUnitLayer(input_dim, output_dim, **params))
        for _ in range(1, self.n_glu):
            self.glu_layers.append(GatedLinearUnitLayer(output_dim, output_dim, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        x = self.glu_layers[0](x)
        layers_left = range(1, self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        prediction_dim=8,
        attention_dim=8,
        n_steps=3,
        gamma=1.3,
        n_glu=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.prediction_dim = prediction_dim
        self.attention_dim = attention_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_glu = n_glu
        self.virtual_batch_size = virtual_batch_size
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)
        self.attention_dim = self.input_dim

        self.initial_splitter = FeatureTransformer(
            self.input_dim,
            prediction_dim + attention_dim,
            n_glu=self.n_glu,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for _ in range(n_steps):
            transformer = FeatureTransformer(
                self.input_dim,
                prediction_dim + attention_dim,
                n_glu=self.n_glu,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                attention_dim,
                self.attention_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        bs = x.shape[0]  # batch size
        if prior is None:
            prior = torch.ones((bs, self.attention_dim)).to(x.device)

        M_loss = 0
        att = self.initial_splitter(x)[:, self.prediction_dim :]
        steps_output = []
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1))
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = nn.ReLU()(out[:, : self.prediction_dim])
            steps_output.append(d)
            # update attention
            att = out[:, self.prediction_dim :]

        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)
        bs = x.shape[0]  # batch size
        prior = torch.ones((bs, self.attention_dim)).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_feature_level = M
            masks[step] = M_feature_level
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = nn.ReLU()(out[:, : self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M_feature_level, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d :]

        return M_explain, masks


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        prediction_dim=8,
        n_steps=3,
        n_glu=1,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.prediction_dim = prediction_dim
        self.n_steps = n_steps
        self.n_glu = n_glu
        self.virtual_batch_size = virtual_batch_size

        self.feat_transformers = nn.ModuleList()

        for _ in range(n_steps):
            transformer = FeatureTransformer(
                prediction_dim,
                prediction_dim,
                n_glu=self.n_glu,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)

        self.reconstruction_layer = nn.Linear(prediction_dim, self.input_dim, bias=False)

    def forward(self, steps_output):
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feat_transformers[step_nb](step_output)
            res = torch.add(res, x)
        res = self.reconstruction_layer(res)
        return res


class GatedLinearUnitLayer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super().__init__()

        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)

        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out


class RandomObfuscator(torch.nn.Module):
    """
    Create and applies obfuscation masks.
    The obfuscation is done at group level to match attention.
    """

    def __init__(self, pretraining_ratio):
        super().__init__()
        self.pretraining_ratio = pretraining_ratio
        self.num_groups = 1

    def forward(self, x):
        bs = x.shape[0]

        obfuscated_vars = torch.bernoulli(self.pretraining_ratio * torch.ones((bs, 1), device=x.device))
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_vars


class TabNetPretraining(nn.Module):
    def __init__(
        self,
        input_dim,
        pretraining_ratio=0.2,
        prediction_dim=8,
        attention_dim=8,
        n_steps=3,
        gamma=1.3,
        n_glu=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        n_decoder=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.prediction_dim = prediction_dim
        self.attention_dim = attention_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_glu = n_glu
        self.pretraining_ratio = pretraining_ratio
        self.n_decoder = n_decoder
        self.virtual_batch_size = virtual_batch_size

        self.masker = RandomObfuscator(self.pretraining_ratio)
        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=input_dim,
            prediction_dim=prediction_dim,
            attention_dim=attention_dim,
            n_steps=n_steps,
            gamma=gamma,
            n_glu=n_glu,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )
        self.decoder = Decoder(
            input_dim=input_dim,
            prediction_dim=prediction_dim,
            n_steps=n_steps,
            n_glu=n_glu,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    def compute_loss(self, y_pred, embedded_x, obf_vars, eps=1e-9):
        errors = y_pred - embedded_x
        reconstruction_errors = torch.mul(errors, obf_vars) ** 2
        batch_means = torch.mean(embedded_x, dim=0)
        batch_means[batch_means == 0] = 1

        batch_stds = torch.std(embedded_x, dim=0) ** 2
        batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
        features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
        # compute the number of obfuscated variables to reconstruct
        nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
        # take the mean of the reconstructed variable errors
        features_loss = features_loss / (nb_reconstructed_variables + eps)
        # here we take the mean per batch, contrary to the paper
        loss = torch.mean(features_loss)
        return loss
        # return self.loss_fn(output, embedded_x, obf_vars)

    def forward(self, x):
        if self.training:
            masked_x, obfuscated_vars = self.masker(x)
            # set prior of encoder with obfuscated groups
            prior = 1 - obfuscated_vars
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, x, obfuscated_vars
        else:
            steps_out, _ = self.encoder(x)
            res = self.decoder(steps_out)
            return res, x, torch.ones(x.shape).to(x.device)

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)
