# FieldawareFactorizationMachineClassifier

> `crank_ml.factorization_machine.fieldaware_factorization_machine_classifier.FieldawareFactorizationMachineClassifier`

Fieldaware Factorization Machine for classification.

$$
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}  + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_{j, f_{j'}}, \mathbf{v}_{j', f_j} \rangle x_{j} x_{j'}
$$

Where $\mathbf{v}_{j, f_{j'}}$ is the latent vector corresponding to $j$ feature for $f_{j'}$ field, and $\mathbf{v}_{j', f_j}$ is the latent vector corresponding to $j'$ feature for $f_j$ field. 

# Parameters

| Parameter     | Description                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `n_features`  | Number of input features                                                                              |
| `embed_dim`   | (_Default_: `64`) Embedding dimension for the latent variables                                        |
| `n_latent_factors` | (_Default_: `10`) The number of latent factors                                                   |
| `n_classes`   | (_Default_: `2`) The number of classes in the classification problem                                  |
| `penalty`     | (_Default_: `l2`) The penalty (aka regularization term) to be used. Defaults to 'l2' which is the standard regularizer for linear SVM models. 'l1' and 'elasticnet' might bring sparsity to the model (feature selection) not achievable with 'l2'. No penalty is added when set to `None``. |
| `alpha`       | (_Default_: `0.0001`) Constant that multiplies the regularization term. The higher the value, the stronger the regularization. |
| `l1_ratio`    | (_Default_: `0.15`) The Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`. `l1_ratio=0` corresponds to L2 penalty, `l1_ratio=1` to L1. Only used if penalty is 'elasticnet'. |

