import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Cross_Modal_Triplet_Loss(torch.nn.Module):
    def __init__(self,
                margin: float = 0.2,
    ):
        super(Cross_Modal_Triplet_Loss, self).__init__()
        self.margin = margin

    def forward(self, pos_shape_feat, img_feat, neg_shape_feat):
        # Using cosine distance
        pos_cosine_similarity = F.cosine_similarity(pos_shape_feat, img_feat, dim=1, eps=1e-6)
        pos_cosine_distance = 1 - pos_cosine_similarity

        neg_cosine_similarity = F.cosine_similarity(neg_shape_feat, img_feat, dim=1, eps=1e-6)
        neg_cosine_distance = 1 - neg_cosine_similarity
        
        # pos_euclidean_distance = F.pairwise_distance(pos_shape_feat, img_feat, keepdim = True)
        # neg_euclidean_distance = F.pairwise_distance(neg_shape_feat, img_feat, keepdim = True)

        ## Adding eps in torch.pow to avoid nan values (Somehow this worlks, but donno why?)
        # cross_modal_triplet_loss = torch.mean(torch.max(pos_cosine_distance**2 - neg_cosine_distance**2 + self.margin, torch.tensor(0.0).cuda()))

        cross_modal_triplet_loss = torch.mean(pos_cosine_distance**2)*0.5 + torch.mean(torch.max(self.margin - neg_cosine_distance**2, torch.tensor(0.0).cuda()))*0.5

        return cross_modal_triplet_loss
