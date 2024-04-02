import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Feature_Distance_Loss(torch.nn.Module):
    def __init__(self):
        super(Feature_Distance_Loss, self).__init__()

    def forward(self, domain_specific_feat, domain_invariant_feat):
        # Normalizing the features
        # pos_shape_feat = F.normalize(pos_shape_feat, p=2, dim=1, eps=1e-6)
        # img_feat = F.normalize(img_feat, p=2, dim=1, eps=1e-6)
        # neg_shape_feat = F.normalize(neg_shape_feat, p=2, dim=1, eps=1e-6)
        domain_specific_feat = F.normalize(domain_specific_feat, p=2, dim=1, eps=1e-6)
        domain_invariant_feat = F.normalize(domain_invariant_feat, p=2, dim=1, eps=1e-6)

        # Inner product
        fdl_loss = torch.mean(torch.bmm(domain_specific_feat.unsqueeze(1), domain_invariant_feat.unsqueeze(2)).squeeze(2).squeeze(1))

        # Square the loss
        fdl_loss = torch.pow(fdl_loss, 2)

        return fdl_loss
