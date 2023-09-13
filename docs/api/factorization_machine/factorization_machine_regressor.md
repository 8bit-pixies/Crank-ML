# FactorizationMachineRegressor

> `crank_ml.factorization_machine.factorization_machine_regressor.FactorizationMachineRegressor`

Factorization Machine for regression.

$$
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}  + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_{j} x_{j'}
$$

Where $\mathbf{v}_j$ and $\mathbf{v}_{j'}$ are $j$ and $j'$ latent vectors respectively. 

# Parameters

| Parameter     | Description                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `n_features`  | Number of input features                                                                              |
| `embed_dim`   | (_Default_: `64`) Embedding dimension for the latent variables                                                |
| `penalty`     | (_Default_: `l2`) The penalty (aka regularization term) to be used. Defaults to 'l2' which is the standard regularizer for linear SVM models. 'l1' and 'elasticnet' might bring sparsity to the model (feature selection) not achievable with 'l2'. No penalty is added when set to `None``. |
| `alpha`       | (_Default_: `0.0001`) Constant that multiplies the regularization term. The higher the value, the stronger the regularization. |
| `l1_ratio`    | (_Default_: `0.15`) The Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`. `l1_ratio=0` corresponds to L2 penalty, `l1_ratio=1` to L1. Only used if penalty is 'elasticnet'. |

