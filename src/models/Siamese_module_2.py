from pyexpat import model
from typing import Any, List
from pandas import concat

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
import torchvision

from src.utils.utils import batch_tensor, unbatch_tensor, imsave

from src.utils.ops import regualarize_rendered_views
from src.models.loss_functions.contrastive_loss import ContrastiveLoss
import torch.nn.functional as F
from src.utils.utils import batch_tensor

class SiameseModule(LightningModule):
    """Example of LightningModule for MNIST classification.

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
        multi_view_net: torch.nn.Module,
        multi_view_renderer: torch.nn.Module,
        mvnet_depth: int,
        feature_extractor_num_layers: int,
        siamese_cnn: torch.nn.Module,
        criterion: torch.nn.Module,
        image_feature_network: torch.nn.Module,
        shape_feature_network: torch.nn.Module,
        image_network_weights: str,
        shape_network_weights: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        # self.net = net
        self.mvtn = multi_view_net  

        self.mvtn_renderer = multi_view_renderer

        depth2featdim = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
        assert mvnet_depth in depth2featdim.keys(), "mvnet_depth must be one of 18, 34, 50, 101, 152"
        # mvnetwork = torchvision.models.__dict__["resnet{}".format(mvnet_depth)](pretrained=True)
    
        # image_feature_extractor = torchvision.models.__dict__["resnet{}".format(mvnet_depth)]()
        image_feature_extractor = image_feature_network

        mvnetwork = shape_feature_network
        image_weights = torch.load(image_network_weights)["state_dict"]
        shape_weights = torch.load(shape_network_weights)["state_dict"]

        # Get rid of the "net." prefix in the weights
        # image_weights = {k[10:]: v for k, v in image_weights.items()}
        image_weights = {k[4:]: v for k, v in image_weights.items()}

        # Get rid of the keys that have 'mvtn' in them
        shape_weights = {k: v for k, v in shape_weights.items() if "mvtn" not in k[:4]}
        shape_weights = {k[4:]: v for k, v in shape_weights.items()}

        # Load the weights
        image_feature_extractor.load_state_dict(image_weights, strict=False)
        mvnetwork.load_state_dict(shape_weights, strict=False)


        self.mvnetwork = torch.nn.Sequential(*list(mvnetwork.children()))
        # self.image_feature_extractor = torch.nn.Sequential(*list(image_feature_extractor.children())[:-1])
        # self.image_feature_extractor = torch.nn.Sequential(*list(image_feature_extractor.children))
        self.image_feature_extractor = image_feature_extractor



        ## TODO: remove this line while training
        self.mvtn.requires_grad_(False)
        self.mvtn_renderer.requires_grad_(False)
        # self.mvnetwork.requires_grad_(False)
        # self.image_feature_extractor.requires_grad_(False)

        self.siamese_cnn = siamese_cnn

        # loss function
        self.criterion = criterion

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_loss = ContrastiveLoss()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_loss_best = MinMetric()

    def forward(self, meshes: torch.Tensor, points: torch.Tensor, image: torch.Tensor):
        c_batch_size = len(meshes)

        with torch.no_grad():
            azim, elev, dist = self.mvtn(points, c_batch_size=c_batch_size)
            rendered_images, _ = self.mvtn_renderer(meshes, points, azim=azim, elev=elev, dist=dist)
            rendered_images = regualarize_rendered_views(rendered_images, 0.0, False, 0.3)

        # 2048 dimensional shape features
        B, M, C, H, W = rendered_images.shape
        input = batch_tensor(rendered_images, dim=1,squeeze=True)
        input = input.type(torch.cuda.FloatTensor)
        shape_features = self.mvnetwork(input)
        shape_features = shape_features.squeeze()

        shape_features = unbatch_tensor(shape_features, B, dim=1, unsqueeze=True)
        shape_features = torch.max(shape_features, dim=1)[0]

        # # 2048 dimensional image features
        image_features = self.image_feature_extractor(image)
            
        # TODO: remove this line while training
        # shape_features = shape_features.unsqueeze(0)

        siamese_feature_shape, siamese_feature_image = self.siamese_cnn(shape_features, image_features)

        return siamese_feature_shape, siamese_feature_image


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def step(self, batch: Any):
        (model_shape, image, label) = batch
        meshes, points = model_shape

        siamese_feature_shape, siamese_feature_image = self.forward(meshes, points, image)

        loss = self.criterion(output1=siamese_feature_shape, output2=siamese_feature_image, label=label)

        # Calculate similarity using cosine similarity
        similarity = F.cosine_similarity(siamese_feature_shape, siamese_feature_image, dim=1)

        # Check if the similarity is above the threshold
        pred = (similarity < 0.5)

        return loss, pred.int(), label.int().squeeze()

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred, label = self.step(batch) 

        # log train metrics
        acc = self.train_acc(pred, label)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "pred": pred, "label": label}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred, label = self.step(batch)

        # log val metrics
        acc = self.val_acc(pred, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": pred, "targets": label}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        # loss = self.val_loss.forward()  # get val loss from current epoch
        # self.val_loss_best.update(loss)  # update best val loss
        # self.log("val/loss_best", self.val_loss_best.compute(), on_epoch=True, prog_bar=True)
        # self.val_loss.reset()
        

    def test_step(self, batch: Any, batch_idx: int):
        # loss = self.step(batch)

        # # log test metrics
        # # acc = self.test_acc(preds, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True)
        # # self.log("test/acc", acc, on_step=False, on_epoch=True)

        # return {"loss": loss} #, "preds": preds, "targets": targets}

        if batch_idx == 0:
            # We check the dissimilarity between the first image in the batch and the rest of the shapes
            (model_shape, image, label) = batch
            mesh, point = model_shape
            batch_size = len(mesh)
            image0 = image[0]

            for i in range(batch_size):
                meshes = mesh[i]
                points = point[i]
                points = points.unsqueeze(0)
                azim, elev, dist = self.mvtn(points, c_batch_size=1)
                rendered_images, _ = self.mvtn_renderer(meshes, points, azim=azim, elev=elev, dist=dist)
                rendered_images = regualarize_rendered_views(rendered_images, 0.0, False, 0.3)

                # Take one of the rendered images and compare it to the first image
                image_i = rendered_images[0][3]
                image0 = image0.squeeze(0)
                concat_image = torch.cat((image0, image_i.detach().cpu()), axis=1)

                # Calculate the dissimilarity between the first image and the rest of the shapes
                # Reshape image0 as 1x3xHxW
                image0 = image0.unsqueeze(0)

                shape_features, image_features = self.forward(meshes, points, image0)
                cosine_similarity = F.cosine_similarity(shape_features, image_features)
                cosine_distance = (1 - cosine_similarity)*100

                # Save the dissimilarity and the image
                imsave(torchvision.utils.make_grid(concat_image), 'results/pretrained_features/image_' + str(i) + f'Dissimilarity: {cosine_distance.item():.2f}'  +  '.png')

        return {"loss": 0} #, "preds": preds, "targets": targets}


    def test_epoch_end(self, outputs: List[Any]):
        # self.test_acc.reset()
        pass

    def configure_optimizers(self):
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
