import torch
from torch import nn


class KMeans(nn.Module):
    def __init__(self, n_features: int, n_clusters=8, halflife=0.5):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.halflife = halflife
        self.centroids = nn.ParameterList(
            [torch.nn.Parameter(torch.rand(self.n_features).float(), requires_grad=False) for _ in range(n_clusters)]
        )

    def forward(self, X):
        batch_size = X.size(0)

        # pad and tile the centroids to use `torch.cdist` api
        all_centroids = torch.zeros([self.n_clusters, self.n_features])
        for indx, centroid in enumerate(self.centroids):
            all_centroids[indx, :] = centroid
        all_centroids = torch.tile(all_centroids, (batch_size, 1, 1))

        # find the closest centroid, and return the index
        predict_output = torch.argmin(torch.cdist(X.unsqueeze(1), all_centroids).squeeze(1), axis=1)

        if self.training:
            # only update the closest centroid
            for indx, centroid in enumerate(self.centroids):
                if torch.sum(predict_output == indx) > 0:
                    centroid = centroid - self.halflife * (centroid - torch.mean(X[predict_output == indx, :], axis=0))

        return predict_output
