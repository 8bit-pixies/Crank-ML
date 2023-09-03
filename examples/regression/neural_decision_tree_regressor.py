import lightning.pytorch as pl
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from crank_ml.tree.neural_decision_tree_regressor import NeuralDecisionTreeRegressor


class NDTreeRegressor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = NeuralDecisionTreeRegressor(100)

    def forward(self, X):
        return self.model(X).float()

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        loss = loss + self.model.penalty()
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
        x, y = batch
        y_hat = self.forward(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


n_samples = 1000
X, y = make_regression(n_samples=n_samples, noise=0.2, random_state=0)
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
model = NDTreeRegressor()
trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
trainer.test(dataloaders=train_data)
trainer.test(dataloaders=val_data)
trainer.test(dataloaders=test_data)
