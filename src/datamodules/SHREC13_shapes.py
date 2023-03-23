from typing import Any, Dict, Optional, Tuple
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from src.datamodules.components.Siamese_SHREC_datamodule import Siamese_SHREC13, collate_fn
from src.datamodules.components.SHREC13_shapes import SHREC13_Shapes


class SHREC13_Shapes_Module(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_mean: Tuple[float, float, float] = [0.8808, 0.8808, 0.9496],
        data_std: Tuple[float, float, float] = [0.3239, 0.3239, 0.1411],
        nb_points: int = 2048,
        dset_norm: int = 2,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([transforms.Resize
                                                ((224, 224)), 
                                             transforms.ToTensor()])

                                             

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 90

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            # trainset = Siamese_SHREC13(self.hparams.data_dir, split="train", nb_points=self.hparams.nb_points, load_textures=False,dset_norm=self.hparams.dset_norm, simplified_mesh=False, transform=self.transforms)

            # testset = Siamese_SHREC13(self.hparams.data_dir, split="test", nb_points=self.hparams.nb_points, load_textures=False,dset_norm=self.hparams.dset_norm, simplified_mesh=False, transform=self.transforms)

            train_dir = os.path.join(self.hparams.data_dir, "train", "model")
            test_dir = os.path.join(self.hparams.data_dir, "test", "model")


            trainset = SHREC13_Shapes(train_dir, split="train", nb_points=self.hparams.nb_points, load_textures=False,dset_norm=self.hparams.dset_norm, simplified_mesh=False, transform=self.transforms)

            testset = SHREC13_Shapes(test_dir, split="test", nb_points=self.hparams.nb_points, load_textures=False,dset_norm=self.hparams.dset_norm, simplified_mesh=False, transform=self.transforms)

            train_size = int(0.8 * len(trainset))
            val_size = len(trainset) - train_size

            self.data_train, self.data_val = random_split(trainset, [train_size, val_size])

            self.data_test = testset


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
