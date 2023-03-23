import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self,
                margin: float = 2.0,
                ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # output1 = output1/ output1.norm(p=2, dim=1, keepdim=True)
      # output2 = output2/ output2.norm(p=2, dim=1, keepdim=True)

      # Calculate the euclidian distance and calculate the contrastive loss
      # euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
      print(torch.flatten(label).cpu().numpy())

      # TODO: look at cosine distance
      # Find cosine distance between the two vectors
      cosine_similarity = F.cosine_similarity(output1, output2, dim=1, eps=1e-6)
      cosine_distance = 1 - cosine_similarity

      # Print cosine distance with 2 decimal places
      print(torch.flatten(cosine_distance).detach().cpu().numpy().round(2))

      # # Calculate the contrastive loss
      # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
      #                             (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      loss_contrastive = torch.sum((1-label) * torch.pow(cosine_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))

    
      # loss_contrastive = torch.mean( (1- label) * cosine_distance +
                                      # (label) * torch.clamp(self.margin - cosine_distance, min=0.0))

      return loss_contrastive
