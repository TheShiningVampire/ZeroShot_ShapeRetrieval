from pyexpat import model
from turtle import pos
from typing import Any, List
from pandas import concat
import numpy as np

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

class DomainDisentangledModule(LightningModule):
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
        domain_disentagled_image_feat: torch.nn.Module,
        domain_disentagled_shape_feat: torch.nn.Module,
        domain_disentangled_semantic_encoder: torch.nn.Module,
        cross_modal_latent_loss: torch.nn.Module,
        cross_modal_triplet_loss: torch.nn.Module,
        cross_modal_classifer_loss: torch.nn.Module,
        image_feature_network: torch.nn.Module,
        shape_feature_network: torch.nn.Module,
        image_network_weights: str,
        shape_network_weights: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        lambda1: float,
        lambda2: float,
        lambda3: float,
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

        self.domain_disentagled_image_feat = domain_disentagled_image_feat
        self.domain_disentagled_shape_feat = domain_disentagled_shape_feat
        self.domain_disentangled_semantic_encoder = domain_disentangled_semantic_encoder

        # loss function
        self.cross_modal_latent_loss = cross_modal_latent_loss
        self.cross_modal_triplet_loss = cross_modal_triplet_loss
        self.cross_modal_classifer_loss = cross_modal_classifer_loss

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_loss = ContrastiveLoss()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_loss_best = MinMetric()

    def forward(self, mesh_positive, points_positive, w2vec_positive, image, mesh_negative, points_negative, w2vec_negative):
        c_batch_size = len(mesh_positive)

        with torch.no_grad():
            azim_p, elev_p, dist_p = self.mvtn(points_positive, c_batch_size=c_batch_size)
            rendered_images_p, _ = self.mvtn_renderer(mesh_positive, points_positive, azim=azim_p, elev=elev_p, dist=dist_p)
            rendered_images_p = regualarize_rendered_views(rendered_images_p, 0.0, False, 0.3)

            azim_n, elev_n, dist_n = self.mvtn(points_negative, c_batch_size=c_batch_size)
            rendered_images_n, _ = self.mvtn_renderer(mesh_negative, points_negative, azim=azim_n, elev=elev_n, dist=dist_n)
            rendered_images_n = regualarize_rendered_views(rendered_images_n, 0.0, False, 0.3)

        # 2048 dimensional shape features for positive model
        B, M, C, H, W = rendered_images_p.shape
        input_p = batch_tensor(rendered_images_p, dim=1,squeeze=True)
        input_p = input_p.type(torch.cuda.FloatTensor)
        shape_features_p = self.mvnetwork(input_p)
        shape_features_p = shape_features_p.squeeze()

        shape_features_p = unbatch_tensor(shape_features_p, B, dim=1, unsqueeze=True)
        shape_features_p = torch.max(shape_features_p, dim=1)[0]


        # 2048 dimensional shape features for negative model
        B, M, C, H, W = rendered_images_n.shape
        input_n = batch_tensor(rendered_images_n, dim=1,squeeze=True)
        input_n = input_n.type(torch.cuda.FloatTensor)
        shape_features_n = self.mvnetwork(input_n)
        shape_features_n = shape_features_n.squeeze()

        shape_features_n = unbatch_tensor(shape_features_n, B, dim=1, unsqueeze=True)
        shape_features_n = torch.max(shape_features_n, dim=1)[0]

        # # 2048 dimensional image features
        image_features = self.image_feature_extractor(image)
            
        # TODO: remove this line while training
        # shape_features = shape_features.unsqueeze(0)

        pos_model_domain_specific, pos_model_domain_inv = self.domain_disentagled_shape_feat(shape_features_p)
        neg_model_domain_specific, neg_model_domain_inv = self.domain_disentagled_shape_feat(shape_features_n)
        image_domain_specific, image_domain_inv = self.domain_disentagled_image_feat(image_features)
        semantic_features = self.domain_disentangled_semantic_encoder(w2vec_positive)


        return (pos_model_domain_specific, pos_model_domain_inv), \
                (image_domain_specific, image_domain_inv), \
                (neg_model_domain_specific, neg_model_domain_inv), \
                semantic_features


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def step(self, batch: Any):
        (positive_model, image, negative_model) = batch
        
        mesh_positive, points_positive, label_positive, w2vec_positive = positive_model
        mesh_negative, points_negative, label_negative, w2vec_negative = negative_model

        pos_model_feat, img_feat, neg_model_feat, semantic_feat = self.forward(mesh_positive, points_positive, w2vec_positive, image, mesh_negative, points_negative, w2vec_negative)

        pos_model_domain_specific, pos_model_domain_inv = pos_model_feat
        image_domain_specific, image_domain_inv = img_feat
        neg_model_domain_specific, neg_model_domain_inv = neg_model_feat
        
        cmd = self.cross_modal_latent_loss(pos_model_domain_specific, image_domain_specific, semantic_feat)

        cmtr = self.cross_modal_triplet_loss(pos_model_domain_specific, image_domain_specific, neg_model_domain_specific)
        
        cmcl1 = self.cross_modal_classifer_loss(pos_model_domain_inv, label_positive)

        cmcl2 = self.cross_modal_classifer_loss(image_domain_inv, label_positive)

        cmcl3 = self.cross_modal_classifer_loss(neg_model_domain_inv, label_negative)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print("cmd: ", cmd.detach().cpu().numpy(),
          "cmtr: ", cmtr.detach().cpu().numpy(),
          "cmcl1: ", cmcl1.detach().cpu().numpy(), 
          "cmcl2: ", cmcl2.detach().cpu().numpy(), 
          "cmcl3: ", cmcl3.detach().cpu().numpy()
          )

        loss = (self.hparams.lambda1* cmd) +\
                (self.hparams.lambda2* cmtr) +\
                (self.hparams.lambda3* (cmcl1 + cmcl2 + cmcl3))

        # Calculate similarity using cosine similarity between the image and positive model features
        similarity_pos = F.cosine_similarity(pos_model_domain_specific, image_domain_specific, dim=1)

        # Calculate similarity using cosine similarity between the image and negative model features
        similarity_neg = F.cosine_similarity(neg_model_domain_specific, image_domain_specific, dim=1)

        # # Print the negative domain specific features
        # print("Negative domain specific features: ", neg_model_domain_specific.detach().cpu().numpy())

        # ## Positive similarity
        # print("similarity_pos: ", similarity_pos.detach().cpu().numpy())

        # ## Print the similarity values for negative models
        # print("similarity_neg: ", similarity_neg.detach().cpu().numpy())        

        # Check if the similarity is above the threshold
        pred_pos = (similarity_pos < 0.5)

        # Ground truth is that the image is from the same domain as the positive model
        gt_pos = torch.zeros_like(pred_pos)

        # For negative model
        pred_neg = (similarity_neg < 0.5)
        gt_neg = torch.ones_like(pred_neg)

        # Print prediction and ground truth for negative model
        print("pred_neg: ", pred_neg.detach().cpu().numpy())
        print("gt_neg: ", gt_neg.detach().cpu().numpy())

        return loss, pred_pos, gt_pos, pred_neg, gt_neg

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred_pos, gt_pos, pred_neg, gt_neg = self.step(batch) 

        # log train metrics
        acc = (self.train_acc(pred_pos, gt_pos) + self.train_acc(pred_neg, gt_neg)) / 2
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "pred_positive": pred_pos, "label_positive": gt_pos, "pred_negative": pred_neg, "label_negative": gt_neg}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_pos, gt_pos, pred_neg, gt_neg = self.step(batch) 

        # log val metrics
        acc = (self.train_acc(pred_pos, gt_pos) + self.train_acc(pred_neg, gt_neg)) / 2

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "pred_positive": pred_pos, "label_positive": gt_pos, "pred_negative": pred_neg, "label_negative": gt_neg}

    def validation_epoch_end(self, outputs: List[Any]):
        # acc = self.val_acc.compute()  # get val accuracy from current epoch
        # loss = self.val_loss.forward()  # get val loss from current epoch
        # self.val_loss_best.update(loss)  # update best val loss
        # self.log("val/loss_best", self.val_loss_best.compute(), on_epoch=True, prog_bar=True)
        # self.val_loss.reset()
        pass
        

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
