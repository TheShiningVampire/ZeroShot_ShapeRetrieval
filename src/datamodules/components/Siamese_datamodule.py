from typing import Dict, List, Optional, Tuple
from pathlib import Path
from os import path
from unicodedata import name
import warnings
import json
from cv2 import transform         #TODO: Uncomment this line
import numpy as np
import glob
import h5py
import pandas as pd
import collections
from torch.utils.data.dataset import Dataset
import os
import torch
from PIL import Image
from src.utils.utils import torch_center_and_normalize, sort_jointly, load_obj, load_text
# from torch._six import container_abcs, string_classes, int_classes

import trimesh
import math
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures

from src.utils.utils import rotation_matrix

import torchvision.datasets as datasets
import random



def collate_fn(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:

            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'pytorch3d.structures.meshes':
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, (int)):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):

        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]


class ShapeNetBase(torch.utils.data.Dataset):
    """
    'ShapeNetBase' implements a base Dataset for ShapeNet and R2N2 with helper methods.
    It is not intended to be used on its own as a Dataset for a Dataloader. Both __init__
    and __getitem__ need to be implemented.
    """

    def __init__(self):
        """
        Set up lists of synset_ids and model_ids.
        """
        self.synset_ids = []
        self.model_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_models = {}
        self.shapenet_dir = ""
        self.model_dir = "model.obj"
        self.load_textures = True
        self.texture_resolution = 4

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.model_ids)

    def __getitem__(self, idx) -> Dict:
        """
        Read a model by the given index. Need to be implemented for every child class
        of ShapeNetBase.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary containing information about the model.
        """
        raise NotImplementedError(
            "__getitem__ should be implemented in the child class of ShapeNetBase"
        )

    def _get_item_ids(self, idx) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - synset_id (str): synset id
            - model_id (str): model id
        """
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        return model

    def _load_mesh(self, model_path) -> Tuple:
        from pytorch3d.io import load_obj

        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=self.load_textures,
            load_textures=self.load_textures,
            texture_atlas_size=self.texture_resolution,
        )
        if self.load_textures:
            textures = aux.texture_atlas

        else:
            textures = verts.new_ones(
                faces.verts_idx.shape[0],
                self.texture_resolution,
                self.texture_resolution,
                3,
            )

        return verts, faces.verts_idx, textures

    def _load_mesh_off(self, model_path) -> Tuple:
        from pytorch3d.io import IO 

        mesh = IO().load_mesh(model_path)
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()

        textures = verts.new_ones(
            faces.shape[0],
            self.texture_resolution,
            self.texture_resolution,
            3,
        )

        return verts, faces, textures


class ShapeNetCore(ShapeNetBase):
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
        self,
        data_dir,
        split,
        nb_points,
        synsets=None,
        version: int = 2,
        load_textures: bool = False,
        texture_resolution: int = 4,
        dset_norm: str = "inf",
        simplified_mesh=False
    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.nb_points = nb_points
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.dset_norm = dset_norm
        self.split = split
        self.simplified_mesh = simplified_mesh

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"
        if self.simplified_mesh:
            self.model_dir = "models/model_normalized_SMPLER.obj"
        splits = pd.read_csv(os.path.join(
            self.shapenet_dir, "shapenet_split.csv"), sep=",", dtype=str)

        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(path.join(self.shapenet_dir, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)

        self.synset_inv = {label: offset for offset,
                           label in self.synset_dict.items()}

        if synsets is not None:

            synset_set = set()
            for synset in synsets:
                if (synset in self.synset_dict.keys()) and (
                    path.isdir(path.join(data_dir, synset))
                ):
                    synset_set.add(synset)
                elif (synset in self.synset_inv.keys()) and (
                    (path.isdir(path.join(data_dir, self.synset_inv[synset])))
                ):
                    synset_set.add(self.synset_inv[synset])
                else:
                    msg = (
                        "Synset category %s either not part of ShapeNetCore dataset "
                        "or cannot be found in %s."
                    ) % (synset, data_dir)
                    warnings.warn(msg)

        else:
            synset_set = {
                synset
                for synset in os.listdir(data_dir)
                if path.isdir(path.join(data_dir, synset))
                and synset in self.synset_dict
            }

        synset_not_present = set(
            self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset])
         for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, self.model_dir)):
                    msg = (
                        "Object file not found in the model directory %s "
                        "under synset directory %s."
                    ) % (model, synset)

                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        self.model_ids, self.synset_ids = sort_jointly(
            [self.model_ids, self.synset_ids], dim=0)
        self.classes = sorted(list(self.synset_inv.keys()))
        self.label_by_number = {k: v for v, k in enumerate(self.classes)}

        split_model_ids, split_synset_ids = [], []
        for ii, model in enumerate(self.model_ids):
            found = splits[splits.modelId.isin([model])]["split"]
            if len(found) > 0:
                if found.item() in self.split:
                    split_model_ids.append(model)
                    split_synset_ids.append(self.synset_ids[ii])
        self.model_ids = split_model_ids
        self.synset_ids = split_synset_ids

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        verts, faces, textures = self._load_mesh(model_path)
        label_str = self.synset_dict[model["synset_id"]]

        verts = torch_center_and_normalize(
            verts.to(torch.float), p=self.dset_norm)

        verts_rgb = torch.ones_like(verts)[None]
        textures = Textures(verts_rgb=verts_rgb)
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        points = trimesh.Trimesh(vertices=verts.numpy(
        ), faces=faces.numpy()).sample(self.nb_points, False)
        points = torch.from_numpy(points).to(torch.float)
        points = torch_center_and_normalize(points, p=self.dset_norm)
        return self.label_by_number[label_str], mesh, points


class Siamese_Pix3D(ShapeNetBase):
    def __init__(self, data_dir, split, nb_points, synsets=None, version: int = 2, 
                load_textures: bool = False, texture_resolution: int = 4, dset_norm: str = "inf", simplified_mesh=False, transform=None):
        super().__init__()
        data_dir = path.join(data_dir, split)
        self.shapenet_dir = data_dir
        self.nb_points = nb_points
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.dset_norm = dset_norm
        self.split = split
        self.simplified_mesh = simplified_mesh
        self.model_dir = "model.obj"
        self.pix3d_dir = path.join(data_dir, "model")
        self.image_dir = path.join(data_dir, "img")
        self.transform = transform

        ## Image folder dataset contatining all the classes
        self.imageFolderDataset = datasets.ImageFolder(self.image_dir)

        data_dir = path.join(data_dir, "model")
        
        synset_set = {
                synset
                for synset in os.listdir(data_dir)
                if path.isdir(path.join(data_dir, synset))
            }
        
        self.classes = sorted(list(synset_set))
        self.label_by_number = {k: v for v, k in enumerate(self.classes)}
        
        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, self.model_dir)):
                    msg = (
                        "Object file not found in the model directory %s "
                        "under synset directory %s."
                    ) % (model, synset)

                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        self.model_ids, self.synset_ids = sort_jointly(
            [self.model_ids, self.synset_ids], dim=0)
        
        
    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.pix3d_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        verts, faces, textures = self._load_mesh(model_path)
        label_str = model["synset_id"]

        verts = torch_center_and_normalize(
            verts.to(torch.float), p=self.dset_norm)

        verts_rgb = torch.ones_like(verts)[None]
        textures = Textures(verts_rgb=verts_rgb)
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        points = trimesh.Trimesh(vertices=verts.numpy(
        ), faces=faces.numpy()).sample(self.nb_points, False)
        points = torch.from_numpy(points).to(torch.float)
        points = torch_center_and_normalize(points, p=self.dset_norm)


        ## Also adding the images now
        ## Taking approximately 50% of the images from same class

        class_indx = self.label_by_number[label_str]
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img_tuple = random.choice(self.imageFolderDataset.imgs)
                if img_tuple[1] == class_indx:
                    img = Image.open(img_tuple[0])

                    # Check if image has 3 channels
                    if img.mode == "RGB":
                        break
        else:
            while True:
                img_tuple = random.choice(self.imageFolderDataset.imgs)
                if img_tuple[1] != class_indx:
                    img = Image.open(img_tuple[0])

                    # Check if image has 3 channels
                    if img.mode == "RGB":
                        break

        

        # TODO: Add transforms here
        if self.transform is not None:
            img = self.transform(img)

        return (mesh, points), img, torch.from_numpy(np.array([int(img_tuple[1] != class_indx)], dtype=np.float32))

# if __name__ == "__main__":
#     data_dir = '/home/SharedData/Vinit/pix3d_preprocessed/'
#     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#     dset_train = Siamese_Pix3D(data_dir, split="train", nb_points=2048, load_textures=False,dset_norm=2, simplified_mesh=False, transform=transform)

#     train_loader = DataLoader(dset_train, batch_size=8,
#                           shuffle=True, num_workers=6, collate_fn=collate_fn, drop_last=True)
