# TabNetPretraining

> `crank_ml.impute.tabnet_pretraining.TabNetPretraining`

This implements the unsupervised TabNet encoder-decoder approach to imputing missing values in datasets.

# Example

Notice that in the example below, the label is not used in the loss function or the training step.

```py
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from crank_ml.impute.tabnet_pretraining import TabNetPretraining


class TabNetImputer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TabNetPretraining(4)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, _ = batch
        output, embedded_x, obf_vars = self.model(x)
        loss = self.model.compute_loss(output, embedded_x, obf_vars)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


X, y = load_iris(return_X_y=True)
X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.33, random_state=42)


train_data = DataLoader(
    TensorDataset(torch.from_numpy(X_train).float(), F.one_hot(torch.from_numpy(y_train), num_classes=3).float()),
    batch_size=64,
)
test_data = DataLoader(
    TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test), num_classes=3).float()),
    batch_size=64,
)
val_data = DataLoader(
    TensorDataset(torch.from_numpy(X_val).float(), F.one_hot(torch.from_numpy(y_val), num_classes=3).float()),
    batch_size=64,
)

trainer = pl.Trainer(accelerator="cpu", max_epochs=250)
model = TabNetImputer()
trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
trainer.test(dataloaders=train_data)
trainer.test(dataloaders=val_data)
trainer.test(dataloaders=test_data)

# usage
model.eval()
reconstruction, _, _ = model(torch.from_numpy(X_train).float())
```