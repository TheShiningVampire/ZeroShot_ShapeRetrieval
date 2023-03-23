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


class SHRECBase(torch.utils.data.Dataset):
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
        self.img_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_imgs = {}
        self.shapenet_dir = ""
        self.model_dir = "model.obj"
        self.load_textures = True
        self.texture_resolution = 4

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.img_ids)

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
        model["img_id"] = self.img_ids[idx]
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



class Siamese_SHREC13(SHRECBase):
    def __init__(self, data_dir, split, nb_points, synsets=None, version: int = 2, 
                load_textures: bool = False, texture_resolution: int = 4, dset_norm: str = "inf", simplified_mesh=False, transform=None):
        super().__init__()
        data_dir = path.join(data_dir, split)
        self.data_dir = data_dir
        self.shapenet_dir = data_dir
        self.nb_points = nb_points
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.dset_norm = dset_norm
        self.split = split
        self.simplified_mesh = simplified_mesh
        self.model_dir = path.join(data_dir, "model")
        self.image_dir = path.join(data_dir, "img")
        self.transform = transform

        ## Making synsets using img folder 
        img_dir = path.join(data_dir, "img")
        
        synset_set = {
                synset
                for synset in os.listdir(img_dir)
                if path.isdir(path.join(img_dir, synset))
            }
        
        self.classes = sorted(list(synset_set))
        self.label_by_number = {k: v for v, k in enumerate(self.classes)}
        
        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for img in os.listdir(path.join(img_dir, synset)):
                self.synset_ids.append(synset)
                self.img_ids.append(img)
            img_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_imgs[synset] = img_count
        self.img_ids, self.synset_ids = sort_jointly(
            [self.img_ids, self.synset_ids], dim=0)
        
        
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
        image = self._get_item_ids(idx)
        img_path = path.join(
            self.image_dir, image["synset_id"], image["img_id"]
        )
        img = Image.open(img_path).convert('RGB')

        ## Also adding the models now
        ## Taking approximately 50% of the models from same class
        ## and 50% from different class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            model_class = image["synset_id"]
            model_id = random.choice(os.listdir(path.join(self.model_dir, model_class)))
        else:
            ## Get different class from image["synset_id"]
            model_class = random.choice(list(set(self.classes) - set([image["synset_id"]])))
            model_id = random.choice(os.listdir(path.join(self.model_dir, model_class)))

        model_path = path.join(self.model_dir, model_class, model_id)
        verts, faces, textures = self._load_mesh_off(model_path)

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

        # TODO: Add transforms here
        if self.transform is not None:
            img = self.transform(img)

        # Label is 0 if same class, 1 if different class
        label = torch.from_numpy(np.array([1 - should_get_same_class],dtype=np.float32))
        label = label.to(torch.float)
        label = torch.squeeze(label)

        return (mesh, points), img, label

