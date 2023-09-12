import numpy as np
import torch

from crank_ml.cluster.kmeans import KMeans

X = torch.from_numpy(np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])).float()

kmeans = KMeans(n_clusters=2, n_features=2)

for _ in range(5):
    _ = kmeans(X)

kmeans.eval()
labels = kmeans(X)
