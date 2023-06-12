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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import os

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
        domain_classifier: torch.nn.Module,
        domain_disentangled_semantic_encoder: torch.nn.Module,
        cross_modal_latent_loss: torch.nn.Module,
        cross_modal_triplet_loss: torch.nn.Module,
        # info_nce_loss: torch.nn.Module,
        domain_classifier_loss: torch.nn.Module,
        feature_distance_loss: torch.nn.Module,
        image_feature_network: torch.nn.Module,
        shape_feature_network: torch.nn.Module,
        image_network_weights: str,
        shape_network_weights: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        lambda1: float,
        lambda2: float,
        lambda3: float,
        lambda4: float,
        num_classes: int,
        tsne_path: str,
        plot_tsne: bool,
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
        # self.mvtn.requires_grad_(False)
        # self.mvtn_renderer.requires_grad_(False)
        # self.mvnetwork.requires_grad_(False)
        # self.image_feature_extractor.requires_grad_(False)

        self.domain_disentagled_image_feat = domain_disentagled_image_feat

        self.domain_disentagled_shape_feat = domain_disentagled_shape_feat

        self.domain_disentangled_semantic_encoder = domain_disentangled_semantic_encoder

        self.domain_classifier = domain_classifier

        # loss function
        self.cross_modal_latent_loss = cross_modal_latent_loss
        self.cross_modal_triplet_loss = cross_modal_triplet_loss
        self.domain_classifer_loss = domain_classifier_loss
        self.feature_distance_loss = feature_distance_loss

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_loss = ContrastiveLoss()
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_loss_best = MinMetric()

        # List for storing the features
        self.pos_model_domain_specific = []
        self.image_domain_specific = []

        self.pos_model_domain_inv = []
        self.image_domain_inv = []

        self.class_labels = []

        # Path to save the tSNE plots
        self.tsne_path = tsne_path

        self.plot_tsne = plot_tsne

    def forward(self, mesh_positive, points_positive, w2vec_positive, image, mesh_negative, points_negative, w2vec_negative):
        c_batch_size = len(mesh_positive)

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


        return (pos_model_domain_specific,       
                pos_model_domain_inv), \
                (image_domain_specific, image_domain_inv), \
                (neg_model_domain_specific, neg_model_domain_inv), \
                semantic_features


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def step(self, batch: Any, train: bool = True):
        (positive_model, image, negative_model) = batch
        
        mesh_positive, points_positive, label_positive, w2vec_positive = positive_model

        mesh_negative, points_negative, label_negative, w2vec_negative = negative_model

        pos_model_feat, img_feat, neg_model_feat, semantic_feat = self.forward(mesh_positive, points_positive, w2vec_positive, image, mesh_negative, points_negative, w2vec_negative)

        # pos_model_feat = self.forward(mesh_positive, points_positive, w2vec_positive, image, mesh_negative, points_negative, w2vec_negative)

        pos_model_domain_specific, pos_model_domain_inv = pos_model_feat
        # pos_model_domain_specific, _ = pos_model_feat           # 512 dim. features
        image_domain_specific, image_domain_inv = img_feat
        # image_domain_specific, _ = img_feat                     # 512 dim. features
        neg_model_domain_specific, neg_model_domain_inv = neg_model_feat
        # neg_model_domain_specific, _ = neg_model_feat           # 512 dim. features
        
        cmd = self.cross_modal_latent_loss(pos_model_domain_specific, image_domain_specific, semantic_feat)

        cmtr = self.cross_modal_triplet_loss(pos_model_domain_specific, 
        image_domain_specific, neg_model_domain_specific)
        
        classification_features_p = self.domain_classifier(pos_model_domain_inv)

        classification_features_n = self.domain_classifier(neg_model_domain_inv)

        classification_features_i = self.domain_classifier(image_domain_inv)

        label_p = torch.zeros_like(label_positive)
        label_n = torch.zeros_like(label_negative)
        label_i = torch.ones_like(label_positive)

        cmcl1 = self.domain_classifer_loss(classification_features_p, label_p)

        cmcl2 = self.domain_classifer_loss(classification_features_i, label_i)

        cmcl3 = self.domain_classifer_loss(classification_features_n, label_n)

        # feature distance loss: dot product between domain specific features and domain invariant features
        fdl1 = self.feature_distance_loss(pos_model_domain_specific, pos_model_domain_inv)
        fdl2 = self.feature_distance_loss(image_domain_specific, image_domain_inv)
        fdl3 = self.feature_distance_loss(neg_model_domain_specific, neg_model_domain_inv)


        
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print(
            "cmd: ", cmd.detach().cpu().numpy(),
          "cmtr: ", cmtr.detach().cpu().numpy(),
          "cmcl1: ", cmcl1.detach().cpu().numpy(), 
          "cmcl2: ", cmcl2.detach().cpu().numpy(), 
          "cmcl3: ", cmcl3.detach().cpu().numpy(),
            "fdl1: ", fdl1.detach().cpu().numpy(),
            "fdl2: ", fdl2.detach().cpu().numpy(),
            "fdl3: ", fdl3.detach().cpu().numpy()
          )
        

        loss = (self.hparams.lambda1* cmd) +\
                (self.hparams.lambda2* cmtr) +\
                (self.hparams.lambda3* (cmcl1 + cmcl2 + cmcl3)) +\
                (self.hparams.lambda4* (fdl1 + fdl2 + fdl3))
        

        # Calculate similarity using cosine similarity between the image and positive model features
        similarity_pos = F.cosine_similarity(pos_model_domain_specific, image_domain_specific, dim=1)

        # Calculate similarity using cosine similarity between the image and negative model features
        similarity_neg = F.cosine_similarity(neg_model_domain_specific, image_domain_specific, dim=1)

        # # # Print the negative domain specific features
        # # print("Negative domain specific features: ", neg_model_domain_specific.detach().cpu().numpy())

        # # ## Positive similarity
        # # print("similarity_pos: ", similarity_pos.detach().cpu().numpy())

        # # ## Print the similarity values for negative models
        # # print("similarity_neg: ", similarity_neg.detach().cpu().numpy())        

        # Check if the similarity is above the threshold
        pred_pos = (similarity_pos < 0.5)

        # Ground truth is that the image is from the same domain as the positive model
        gt_pos = torch.zeros_like(pred_pos)

        # For negative model
        pred_neg = (similarity_neg < 0.5)
        gt_neg = torch.ones_like(pred_neg)

        # # # # Print prediction and ground truth for negative model
        # # # print("pred_neg: ", pred_neg.detach().cpu().numpy())
        # # # print("gt_neg: ", gt_neg.detach().cpu().numpy())

        if self.plot_tsne:
            # If training then store features of positive model and image for classes 0-9
            if train:
                valid_indices = (label_positive < 10)
                self.pos_model_domain_specific.append(pos_model_domain_specific[valid_indices].detach().cpu().numpy())
                self.image_domain_specific.append(image_domain_specific[valid_indices].detach().cpu().numpy())
                self.pos_model_domain_inv.append(pos_model_domain_inv[valid_indices].detach().cpu().numpy())
                self.image_domain_inv.append(image_domain_inv[valid_indices].detach().cpu().numpy())
                self.class_labels.append(label_positive[valid_indices].detach().cpu().numpy())

        return loss, pred_pos, gt_pos, pred_neg, gt_neg

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred_pos, gt_pos, pred_neg, gt_neg = self.step(batch, train=True)

        # log train metrics
        acc = (self.train_acc(pred_pos, gt_pos) + self.train_acc(pred_neg, gt_neg)) / 2
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # # we can return here dict with any tensors
        # # and then read it in some callback or in `training_epoch_end()` below
        # # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "pred_positive": pred_pos, "label_positive": gt_pos, "pred_negative": pred_neg, "label_negative": gt_neg}
    
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

        if self.plot_tsne:
            # Find the tSNE embedding of the features
            pos_model_domain_specific_ts= np.concatenate(self.pos_model_domain_specific, axis=0)
            image_domain_specific_ts = np.concatenate(self.image_domain_specific, axis=0)
            pos_model_domain_inv_ts = np.concatenate(self.pos_model_domain_inv, axis=0)
            image_domain_inv_ts = np.concatenate(self.image_domain_inv, axis=0)
            class_labels_ts = np.concatenate(self.class_labels, axis=0)

            tsne = TSNE(n_components=2, random_state=0)
            # pos_model_domain_specific_ts = tsne.fit_transform(pos_model_domain_specific_ts)
            # image_domain_specific_ts = tsne.fit_transform(image_domain_specific_ts)
            # pos_model_domain_inv_ts = tsne.fit_transform(pos_model_domain_inv_ts)
            # image_domain_inv_ts = tsne.fit_transform(image_domain_inv_ts)

            # Concatenate model domain specific and image domain specific features
            domain_specific_ts = np.concatenate((pos_model_domain_specific_ts, image_domain_specific_ts), axis=0)
            num_pos_model = pos_model_domain_specific_ts.shape[0]
            domain_specific_ts_labels = np.concatenate((class_labels_ts, class_labels_ts), axis=0)

            domain_specific_ts = tsne.fit_transform(domain_specific_ts)
            pos_model_domain_specific_ts = domain_specific_ts[:num_pos_model]
            image_domain_specific_ts = domain_specific_ts[num_pos_model:]

            # In the same plot plot the model domain specific features and image domain specific features
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.scatterplot(x=pos_model_domain_specific_ts[:,0], y=pos_model_domain_specific_ts[:,1], hue=class_labels_ts, ax=ax)
            sns.scatterplot(x=image_domain_specific_ts[:,0], y=image_domain_specific_ts[:,1], hue=class_labels_ts, ax=ax, marker='+')

            # # Plot the tSNE embedding
            # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            # sns.scatterplot(x=pos_model_domain_specific_ts[:,0], y=pos_model_domain_specific_ts[:,1], hue=class_labels_ts, ax=ax[0,0])
            # # Use + marker for image domain specific features
            # sns.scatterplot(x=image_domain_specific_ts[:,0], y=image_domain_specific_ts[:,1], hue=class_labels_ts, ax=ax[0,1], marker='+')

            # Save the figure
            save_path = self.tsne_path + 'domain_specific' + str(self.current_epoch) + '.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

            # Clear the lists
            self.pos_model_domain_specific = []
            self.image_domain_specific = []
            self.pos_model_domain_inv = []
            self.image_domain_inv = []


    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_pos, gt_pos, pred_neg, gt_neg = self.step(batch, train=False) 

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
        self.val_acc.reset()
        

    def test_step(self, batch: Any, batch_idx: int):
        # loss = self.step(batch)

        # # log test metrics
        # # acc = self.test_acc(preds, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True)
        # # self.log("test/acc", acc, on_step=False, on_epoch=True)

        # return {"loss": loss} #, "preds": preds, "targets": targets}

        print("Working on test set")
        # We check the dissimilarity between the first image in the batch and the rest of the shapes
        # (model_shape, image, label) = batch
        # mesh, point = model_shape

        (positive_model, image, negative_model) = batch
    
        mesh_positive, points_positive, label_positive, w2vec_positive = positive_model

        mesh_negative, points_negative, label_negative, w2vec_negative = negative_model

        pos_model_feat, img_feat, neg_model_feat, semantic_feat = self.forward(mesh_positive, points_positive, w2vec_positive, image, mesh_negative, points_negative, w2vec_negative)

        # pos_model_feat = self.forward(mesh_positive, points_positive, w2vec_positive, image, mesh_negative, points_negative, w2vec_negative)

        pos_model_domain_specific, pos_model_domain_inv = pos_model_feat
        # pos_model_domain_specific, _ = pos_model_feat           # 512 dim. features
        image_domain_specific, image_domain_inv = img_feat
        # image_domain_specific, _ = img_feat                     # 512 dim. features
        neg_model_domain_specific, neg_model_domain_inv = neg_model_feat
        # neg_model_domain_specific, _ = neg_model_feat           # 512 dim. features

        batch_size = len(mesh_positive)

        mean=[0.9799, 0.9799, 0.9799]
        std=[0.1075, 0.1075, 0.1075]

        for i in range(batch_size):
            img = image[i]
            meshes_p = mesh_positive[i]
            points_p = points_positive[i]

            meshes_n = mesh_negative[i]
            points_n = points_negative[i]
            
            points_p = points_p.unsqueeze(0)
            azim_p, elev_p, dist_p = self.mvtn(points_p, c_batch_size=1)
            rendered_images_p, _ = self.mvtn_renderer(meshes_p, points_p, azim=azim_p, elev=elev_p, dist=dist_p)
            rendered_images_p = regualarize_rendered_views(rendered_images_p, 0.0, False, 0.3)

            points_n = points_n.unsqueeze(0)
            azim_n, elev_n, dist_n = self.mvtn(points_n, c_batch_size=1)
            rendered_images_n, _ = self.mvtn_renderer(meshes_n, points_n, azim=azim_n, elev=elev_n, dist=dist_n)
            rendered_images_n = regualarize_rendered_views(rendered_images_n, 0.0, False, 0.3)

            # Take one of the rendered images and compare it to the first image
            # image_i = rendered_images[0][7]
            # image0 = image0.squeeze(0)
            # image1 = image0*std[0] + mean[0]
            # concat_image = torch.cat((image1, image_i), axis=1)

            image_p = rendered_images_p[0][7]
            image_n = rendered_images_n[0][7]
            img = img.squeeze(0)
            img = img*std[0] + mean[0]
            concat_image = torch.cat((img, image_p, image_n), axis=1)

            img = image.unsqueeze(0)

            # Calculate the dissimilarity between the first image and the rest of the shapes
            # Reshape image0 as 1x3xHxW

            shape_features_p = pos_model_domain_specific[i]
            image_features = image_domain_specific[i]
            shape_features_n = neg_model_domain_specific[i]

            shape_features_p = shape_features_p.unsqueeze(0)
            shape_features_n = shape_features_n.unsqueeze(0)
            image_features = image_features.unsqueeze(0)
            
            cosine_similarity_p = F.cosine_similarity(shape_features_p, image_features)
            cosine_distance_p = (1 - cosine_similarity_p)*100

            cosine_similarity_n = F.cosine_similarity(shape_features_n, image_features)
            cosine_distance_n = (1 - cosine_similarity_n)*100

            # # Find euclidean distance
            euclidean_distance_p = torch.dist(shape_features_p, image_features)
            euclidean_distance_n = torch.dist(shape_features_n, image_features)

            save_path = 'results/complete_model_5/' + str(batch_idx) + '/'

            # if same_class:
            #     save_path = save_path + 'same_class/'
            # else:
            #     save_path = save_path + 'different_class/'

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Make the image grid
            grid = torchvision.utils.make_grid(concat_image)

            grid_np = grid.detach().cpu().numpy().transpose(1,2,0)

            # Plot the image grid and write the dissimilarity values below the rendered images
            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.imshow(grid_np)
            # ax.text(0, 0, 'Dissimilarity: ' + f'{cosine_distance_p.item():.6f}' + ' Euclidean distance: ' + f'{euclidean_distance_p.item():.6f}', fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
            # ax.text(0, 32, 'Dissimilarity: ' + f'{cosine_distance_n.item():.6f}' + ' Euclidean distance: ' + f'{euclidean_distance_n.item():.6f}', fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

            plt.figure(figsize=(10, 10))
            plt.imshow(grid_np)
            plt.text(0, 0, 'Cosine distance: ' + f'{cosine_distance_p.item():.6f}' + ' Euclidean distance: ' + f'{euclidean_distance_p.item():.6f}', fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
            plt.text(0, 32, 'Cosine distance: ' + f'{cosine_distance_n.item():.6f}' + ' Euclidean distance: ' + f'{euclidean_distance_n.item():.6f}', fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

            # Save the plot using plt.savefig()
            plt.savefig(save_path + "image_" + str(i) + "_dissimilarity_" + f'{cosine_distance_p.item():.6f}'  +  '.png')

            # # Save the dissimilarity and the image
            # imsave(torchvision.utils.make_grid(concat_image), save_path + "image_" + str(i) + "_dissimilarity_" + f'{cosine_distance.item():.6f}'  +  '.png')

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
