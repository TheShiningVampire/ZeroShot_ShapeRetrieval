import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Info_NCE_Loss(torch.nn.Module):
    def __init__(self):
        super(Info_NCE_Loss, self).__init__()

    def forward(self, pos_shape_feat, img_feat, neg_shape_feat):
        pos_shape_feat = pos_shape_feat/ (pos_shape_feat.norm(p=2, dim=1, keepdim=True) + 1e-6)
        img_feat = img_feat/ (img_feat.norm(p=2, dim=1, keepdim=True) + 1e-6)
        neg_shape_feat = neg_shape_feat/ (neg_shape_feat.norm(p=2, dim=1, keepdim=True) + 1e-6)

        # # Using cosine distance
        # pos_cosine_similarity = F.cosine_similarity(pos_shape_feat, img_feat, dim=1, eps=1e-6)
        # pos_cosine_distance = 1 - pos_cosine_similarity

        

        # neg_cosine_similarity = F.cosine_similarity(neg_shape_feat, img_feat, dim=1, eps=1e-6)
        # neg_cosine_distance = 1 - neg_cosine_similarity

        # ## Adding eps in torch.pow to avoid nan values (Somehow this worlks, but donno why?)
        # cross_modal_triplet_loss = torch.mean(torch.max(pos_cosine_distance**2 - neg_cosine_distance**2 + self.margin, torch.tensor(0.0).cuda()))

        # num = exp(pos_shape_feat.img_feat)
        num = torch.exp(torch.bmm(pos_shape_feat.unsqueeze(1), img_feat.unsqueeze(2))).squeeze(2).squeeze(1)
        den = num + torch.exp(torch.bmm(neg_shape_feat.unsqueeze(1), img_feat.unsqueeze(2))).squeeze(2).squeeze(1)
        info_nce_loss = -torch.log(num/den)
        info_nce_loss = torch.mean(info_nce_loss)

        return info_nce_loss
