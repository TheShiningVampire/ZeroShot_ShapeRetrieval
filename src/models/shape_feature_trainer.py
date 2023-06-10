from typing import Any, List
from black import out

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
import torchvision

from src.utils.utils import batch_tensor, unbatch_tensor, imsave

from src.utils.ops import regualarize_rendered_views
from src.models.loss_functions.contrastive_loss import ContrastiveLoss
import torch.nn.functional as F

import numpy as np


class SHREC_SHAPE_Feat_Trainer(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        multi_view_net: torch.nn.Module,
        multi_view_renderer: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_classes: int,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.mvtn = multi_view_net  
        self.mvtn_renderer = multi_view_renderer

        # weight_path = "/home/SharedData/Vinit/logs/temp1/checkpoints/epoch_034.ckpt"
        # weights = torch.load(weight_path)["state_dict"]
        
        # # From the keys of the state_dict, remove net. prefix
        # weights = {k[4:]: v for k, v in weights.items()}

        # # Load the weights
        # self.net.load_state_dict(weights)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, meshes: torch.Tensor, points: torch.Tensor):
        c_batch_size = len(meshes)

        azim, elev, dist = self.mvtn(points, c_batch_size=c_batch_size)
        rendered_images, _ = self.mvtn_renderer(meshes, points, azim=azim, elev=elev, dist=dist)
        rendered_images = regualarize_rendered_views(rendered_images, 0.0, False, 0.3)

        return self.net(rendered_images)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        

    def step(self, batch: Any):
        y, meshes, points = batch
        logits = self.forward(meshes, points)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        # print(
        #     "loss: ", loss.detach().cpu().numpy()
        #   )

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
