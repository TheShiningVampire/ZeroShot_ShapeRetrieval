import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Domain_Classification_Loss(torch.nn.Module):
    def __init__(self):
        super(Domain_Classification_Loss, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, feature, label):
        return self.criterion(feature, label)
