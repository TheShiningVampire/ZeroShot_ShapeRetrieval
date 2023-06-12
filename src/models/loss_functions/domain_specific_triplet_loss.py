import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Domain_Specific_Triplet_Loss(torch.nn.Module):
    def __init__(self,
                margin: float = 0.2,
    ):
        super(Domain_Specific_Triplet_Loss, self).__init__()
        self.margin = margin

    def forward(self, pos_shape_feat, img_feat, neg_shape_feat):
        # # Using cosine distance
        # pos_cosine_similarity = F.cosine_similarity(pos_shape_feat, img_feat, dim=1, eps=1e-6)
        # pos_cosine_distance = 1 - pos_cosine_similarity

        # neg_cosine_similarity = F.cosine_similarity(neg_shape_feat, img_feat, dim=1, eps=1e-6)
        # neg_cosine_distance = 1 - neg_cosine_similarity
        

        shape_feat_similarity = F.cosine_similarity(pos_shape_feat, neg_shape_feat, dim=1, eps=1e-6)
        shape_feat_distance = 1 - shape_feat_similarity

        image_feat_similarity = F.cosine_similarity(pos_shape_feat, neg_shape_feat, dim=1, eps=1e-6)

        # cross_modal_triplet_loss = torch.mean(pos_cosine_distance**2)*0.5 + torch.mean(torch.max(self.margin - neg_cosine_distance**2, torch.tensor(0.0).cuda()))*0.5

        # cross_modal_triplet_loss = torch.mean(pos_euclidean_distance**2)*0.5 + torch.mean(torch.max(self.margin - neg_euclidean_distance**2, torch.tensor(0.0).cuda()))*0.5

        domain_specific_triplet_loss = torch.mean(shape_feat_distance**2)*0.5 + torch.mean(torch.max(self.margin - image_feat_similarity**2, torch.tensor(0.0).cuda()))*0.5

        return domain_specific_triplet_loss
