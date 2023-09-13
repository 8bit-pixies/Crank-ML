# FactorizationMachineClassifier

> `crank_ml.factorization_machine.factorization_machine_classifier.FactorizationMachineClassifier`

Factorization Machine for classification.

$$
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}  + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_{j} x_{j'}
$$

Where $\mathbf{v}_j$ and $\mathbf{v}_{j'}$ are $j$ and $j'$ latent vectors respectively. 

# Parameters

| Parameter     | Description                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `n_features`  | Number of input features                                                                              |
| `embed_dim`   | (_Default_: `64`) Embedding dimension for the latent variables                                                |
| `n_classes`   | (_Default_: `2`) The number of classes in the classification problem |
| `penalty`     | (_Default_: `l2`) The penalty (aka regularization term) to be used. Defaults to 'l2' which is the standard regularizer for linear SVM models. 'l1' and 'elasticnet' might bring sparsity to the model (feature selection) not achievable with 'l2'. No penalty is added when set to `None``. |
| `alpha`       | (_Default_: `0.0001`) Constant that multiplies the regularization term. The higher the value, the stronger the regularization. |
| `l1_ratio`    | (_Default_: `0.15`) The Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`. `l1_ratio=0` corresponds to L2 penalty, `l1_ratio=1` to L1. Only used if penalty is 'elasticnet'. |

# Example

```py
import lightning.pytorch as pl
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from crank_ml.factorization_machine.factorization_machine_classifier import FactorizationMachineClassifier


class FMClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FactorizationMachineClassifier(2, n_classes=2)

    def forward(self, X):
        return self.model(X).float()

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics)
        loss = loss + self.model.penalty()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(y_hat, y)
        acc = torch.mean((torch.round(y_hat) == y).float())
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


n_samples = 1000
X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=0)
X_train_, X_test, y_train_, y_test = train_test_split(X, y.reshape(-1, 1), test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.33, random_state=42)


train_data = DataLoader(
    TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
    batch_size=64,
)
test_data = DataLoader(
    TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()),
    batch_size=64,
)
val_data = DataLoader(
    TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
    batch_size=64,
)

trainer = pl.Trainer(accelerator="cpu", max_epochs=100)
model = FMClassifier()
trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
trainer.test(dataloaders=train_data)
trainer.test(dataloaders=val_data)
trainer.test(dataloaders=test_data)
```