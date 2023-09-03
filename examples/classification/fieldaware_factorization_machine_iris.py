import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy

from crank_ml.factorization_machine.factorization_machine_classifier import FactorizationMachineClassifier
from crank_ml.factorization_machine.fieldaware_factorization_machine_classifier import (
    FieldawareFactorizationMachineClassifier,
)


class FFMClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FieldawareFactorizationMachineClassifier(4, embed_dim=4, n_latent_factors=4, n_classes=3)

    def forward(self, X):
        return self.model(X).float()

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
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
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(F.one_hot(torch.argmax(y_hat, axis=1), num_classes=3), y, task="multiclass", num_classes=3)
        return loss, acc

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

trainer = pl.Trainer(accelerator="cpu", max_epochs=100)
model = FFMClassifier()
trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
trainer.test(dataloaders=train_data)
trainer.test(dataloaders=val_data)
trainer.test(dataloaders=test_data)
