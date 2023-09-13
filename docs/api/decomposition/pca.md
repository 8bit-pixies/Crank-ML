# PCA

> `crank_ml.decomposition.pca.PCA`

Performs linear Principal Component Analysis (PCA) on a low-rank matrix, batches of such matrices, or sparse matrix. This approach uses the nearly optimal approximation of a singular value decomposition of a centered matrix.

Our implementation is a wrapper over the `pytorch` implementation with polyak averaging over the weights. As a consequence this may not be stable. 


# Parameters

| Parameter       | Description                                                                      |
| --------------- | -------------------------------------------------------------------------------- |
| `n_features`    | Number of input features                                                         |
| `n_components`  | (_Default_: `6`) Maximum number of components to keep.                           |
| `polyak_weight` | (_Default_: `0.1`) The weight update rate, a reasonable value if between 0 and 1 |

# Example

```py
import numpy as np
import torch

from crank_ml.decomposition.pca import PCA

X = torch.from_numpy(np.array([[1, 2, 3], [1, 4, 5], [1, 0, 1], [10, 2, 2], [10, 4, 3], [10, 0, 1]])).float()

pca = PCA(n_features=3, n_components=2)

for _ in range(5):
    _ = pca(X)

pca.eval()
pca_decomposition = pca(X)
```

