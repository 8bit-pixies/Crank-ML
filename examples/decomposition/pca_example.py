import numpy as np
import torch

from crank_ml.decomposition.pca import PCA

X = torch.from_numpy(np.array([[1, 2, 3], [1, 4, 5], [1, 0, 1], [10, 2, 2], [10, 4, 3], [10, 0, 1]])).float()

pca = PCA(n_features=3, n_components=2)

for _ in range(5):
    _ = pca(X)

pca.eval()
pca_decomposition = pca(X)
