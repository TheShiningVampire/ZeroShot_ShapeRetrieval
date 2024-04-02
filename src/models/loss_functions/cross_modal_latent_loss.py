import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Cross_Modal_Latent_Loss(torch.nn.Module):
    def __init__(self):
        super(Cross_Modal_Latent_Loss, self).__init__()

    def forward(self, output1, output2, w2vec):
      # Calculate the euclidian distance and calculate the contrastive loss
        euclidean_distance_1 = F.pairwise_distance(output1, w2vec, keepdim = True)
        euclidean_distance_2 = F.pairwise_distance(output2, w2vec, keepdim = True)

      # # Calculate the cosine distance and calculate the contrastive loss
      #   cosine_similarity_1 = F.cosine_similarity(output1, w2vec, dim=1, eps=1e-6)
      #   cosine_similarity_2 = F.cosine_similarity(output2, w2vec, dim=1, eps=1e-6)

      #   cosine_distance_1 = 1 - cosine_similarity_1
      #   cosine_distance_2 = 1 - cosine_similarity_2


        cross_modal_latent_loss = torch.mean(torch.pow(euclidean_distance_1, 2) + torch.pow(euclidean_distance_2, 2))
        
        # cross_modal_latent_loss = torch.mean(torch.pow(cosine_distance_1, 2) + torch.pow(cosine_distance_2, 2))

        return cross_modal_latent_loss
