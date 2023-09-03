# Feature Discretization

A demonstration of feature discretization on synthetic classification
datasets. Feature discretization decomposes each feature into a set of
bins, here equally distributed in width. The discrete values are then
one-hot encoded, and given to a linear classifier. This preprocessing
enables a non-linear behavior even though the classifier is linear.

``` python
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy

from crank_ml.linear_model.sgd_classifier import SGDClassifier
from crank_ml.preprocessing.kbins_discretizer import KBinsDiscretizer


class SimpleClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SGDClassifier(2, n_classes=2)

    def forward(self, X):
        return self.model(X)

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
        acc = torch.mean((torch.argmax(y_hat, axis=1) == y).float())
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class DiscretizedClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.kbins = KBinsDiscretizer(n_features=2, encode="onehot-flatten")
        self.model = SGDClassifier(self.kbins.get_output_shape(), n_classes=2)

    def forward(self, X):
        return self.model(self.kbins(X).float())

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


n_samples = 100

datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, random_state=0),
    make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=2,
        n_clusters_per_class=1,
    ),
]

for X, y in datasets:
    train_data = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y.reshape(-1, 1)).float()),
        batch_size=64,
    )

    trainer = pl.Trainer(accelerator="cpu", max_epochs=100)
    model = SimpleClassifier()
    trainer.fit(model, train_dataloaders=train_data)
    trainer.test(dataloaders=train_data)

    trainer = pl.Trainer(accelerator="cpu", max_epochs=100)
    model = DiscretizedClassifier()
    trainer.fit(model, train_dataloaders=train_data)
    trainer.test(dataloaders=train_data)
```

    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/trainer/setup.py:201: UserWarning: MPS available but not used. Set `accelerator` and `devices` using `Trainer(accelerator='mps', devices=1)`.
      rank_zero_warn(
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:67: UserWarning: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
      warning_cache.warn(
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/trainer/configuration_validator.py:71: PossibleUserWarning: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
      rank_zero_warn(

      | Name  | Type          | Params
    ----------------------------------------
    0 | model | SGDClassifier | 3     
    ----------------------------------------
    3         Trainable params
    0         Non-trainable params
    3         Total params
    0.000     Total estimated model params size (MB)
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:438: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 10 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/loops/fit_loop.py:281: PossibleUserWarning: The number of training batches (2) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      rank_zero_warn(
    `Trainer.fit` stopped: `max_epochs=100` reached.
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/checkpoint_connector.py:149: UserWarning: `.test(ckpt_path=None)` was called without a model. The best model of the previous `fit` call will be used. You can pass `.test(ckpt_path='best')` to use the best model or `.test(ckpt_path='last')` to use the last model. If you pass a value, this warning will be silenced.
      rank_zero_warn(
    Restoring states from the checkpoint path at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_0/checkpoints/epoch=99-step=200.ckpt
    Loaded model weights from the checkpoint at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_0/checkpoints/epoch=99-step=200.ckpt
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:438: PossibleUserWarning: The dataloader, test_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 10 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(
    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /Users/csiu/Library/Caches/pypoetry/virtualenvs/crank-ml-xoeVPo1n-py3.10/lib/python3.10/site-packages/lightning/pytorch/trainer/setup.py:201: UserWarning: MPS available but not used. Set `accelerator` and `devices` using `Trainer(accelerator='mps', devices=1)`.
      rank_zero_warn(

      | Name  | Type             | Params
    -------------------------------------------
    0 | kbins | KBinsDiscretizer | 0     
    1 | model | SGDClassifier    | 511   
    -------------------------------------------
    511       Trainable params
    0         Non-trainable params
    511       Total params
    0.002     Total estimated model params size (MB)
    `Trainer.fit` stopped: `max_epochs=100` reached.
    Restoring states from the checkpoint path at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_1/checkpoints/epoch=99-step=200.ckpt
    Loaded model weights from the checkpoint at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_1/checkpoints/epoch=99-step=200.ckpt
    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs

      | Name  | Type          | Params
    ----------------------------------------
    0 | model | SGDClassifier | 3     
    ----------------------------------------
    3         Trainable params
    0         Non-trainable params
    3         Total params
    0.000     Total estimated model params size (MB)
    `Trainer.fit` stopped: `max_epochs=100` reached.
    Restoring states from the checkpoint path at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_2/checkpoints/epoch=99-step=200.ckpt
    Loaded model weights from the checkpoint at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_2/checkpoints/epoch=99-step=200.ckpt
    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs

      | Name  | Type             | Params
    -------------------------------------------
    0 | kbins | KBinsDiscretizer | 0     
    1 | model | SGDClassifier    | 511   
    -------------------------------------------
    511       Trainable params
    0         Non-trainable params
    511       Total params
    0.002     Total estimated model params size (MB)
    `Trainer.fit` stopped: `max_epochs=100` reached.
    Restoring states from the checkpoint path at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_3/checkpoints/epoch=99-step=200.ckpt
    Loaded model weights from the checkpoint at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_3/checkpoints/epoch=99-step=200.ckpt
    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs

      | Name  | Type          | Params
    ----------------------------------------
    0 | model | SGDClassifier | 3     
    ----------------------------------------
    3         Trainable params
    0         Non-trainable params
    3         Total params
    0.000     Total estimated model params size (MB)
    `Trainer.fit` stopped: `max_epochs=100` reached.
    Restoring states from the checkpoint path at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_4/checkpoints/epoch=99-step=200.ckpt
    Loaded model weights from the checkpoint at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_4/checkpoints/epoch=99-step=200.ckpt
    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs

      | Name  | Type             | Params
    -------------------------------------------
    0 | kbins | KBinsDiscretizer | 0     
    1 | model | SGDClassifier    | 511   
    -------------------------------------------
    511       Trainable params
    0         Non-trainable params
    511       Total params
    0.002     Total estimated model params size (MB)
    `Trainer.fit` stopped: `max_epochs=100` reached.
    Restoring states from the checkpoint path at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_5/checkpoints/epoch=99-step=200.ckpt
    Loaded model weights from the checkpoint at /Users/csiu/projects/crank_ml/examples/classification/lightning_logs/version_5/checkpoints/epoch=99-step=200.ckpt

    Training: 0it [00:00, ?it/s]

    Testing: 0it [00:00, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_acc          </span>│<span style="color: #800080; text-decoration-color: #800080">            0.5            </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.3121366798877716     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>

    Training: 0it [00:00, ?it/s]

    Testing: 0it [00:00, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_acc          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.9700000286102295     </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.15714089572429657    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>

    Training: 0it [00:00, ?it/s]

    Testing: 0it [00:00, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_acc          </span>│<span style="color: #800080; text-decoration-color: #800080">            0.5            </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.6925927996635437     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>

    Training: 0it [00:00, ?it/s]

    Testing: 0it [00:00, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_acc          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.8500000238418579     </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.2880871891975403     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>

    Training: 0it [00:00, ?it/s]

    Testing: 0it [00:00, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_acc          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.5099999904632568     </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.20054015517234802    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>

    Training: 0it [00:00, ?it/s]

    Testing: 0it [00:00, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_acc          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.9300000071525574     </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.1860816925764084     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>
