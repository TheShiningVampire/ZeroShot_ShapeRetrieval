import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Maximum_Mean_Discrepancy_Loss(torch.nn.Module):
    def __init__(self):
        super(Maximum_Mean_Discrepancy_Loss, self).__init__()

    def forward(self, output1, output2):
        delta = output1 - output2
        delta = delta.pow(2).mean(1)

        mmd_loss = torch.exp(-delta)

        mmd_loss = torch.mean(mmd_loss)

        return mmd_loss
