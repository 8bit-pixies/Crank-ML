# KMeans

The most common way to implement batch k-means is to use Lloyd's algorithm, which consists in assigning all the data points to a set of cluster centers and then moving the centers accordingly.

In this implementation we start by finding the cluster that is closest to the current observation. We then move the cluster's central position towards the new observation. The halflife parameter determines by how much to move the cluster toward the new observation.

The KMeans implementation does not require learning via differentiation, and is updated analytically.

# Parameters

| Parameter     | Description                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `n_features`  | Number of input features                                                                              |
| `n_clusters`  | (_Default_: `8`) Maximum number of clusters to assign.                                                |
| `halflife`    | (_Default_: `0.5`) Amount by which to move the cluster centers, a reasonable value if between 0 and 1 |

# Example

```py
import numpy as np
import torch

from crank_ml.cluster.kmeans import KMeans

X = torch.from_numpy(np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])).float()

kmeans = KMeans(n_clusters=2, n_features=2)

for _ in range(5):
    _ = kmeans(X)

kmeans.eval()
labels = kmeans(X)
```