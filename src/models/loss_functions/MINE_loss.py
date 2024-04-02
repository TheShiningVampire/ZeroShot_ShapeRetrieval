import torch
import torch.nn as nn
import torch.nn.functional as F

class MINE(nn.Module):
    def __init__(self,
                 feature_dim: int = 300,
                 ):
        super().__init__()
        self.fc1 = nn.Linear(2*feature_dim, 100)  # Adjusted to accept 600-dimensional input
        self.fc2 = nn.Linear(100, 1)
        self.ma_et = None

    #     self.initialize_weights()

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight.data)
    #             nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x, y):
        # Concatenate x and y
        z = torch.cat((x, y), dim=1)
        
        # Feed z into a small neural network
        h = F.relu(self.fc1(z))
        mi_estimate = self.fc2(h)
        
        # Compute the negative expectation of the exp of the output over shuffled samples
        shuffled_idxs = torch.randperm(x.size(0))
        y_shuffled = y[shuffled_idxs]
        z_shuffled = torch.cat((x, y_shuffled), dim=1)
        h_shuffled = F.relu(self.fc1(z_shuffled))
        mi_shuffled_estimate = self.fc2(h_shuffled)
        negative_expectation = torch.mean(torch.exp(mi_shuffled_estimate))
        
        # Track an exponentially weighted moving average of the negative expectation
        # This is used to compute the final loss
        self.ma_et = negative_expectation.detach() if self.ma_et is None else (1 - 0.01) * self.ma_et + 0.01 * negative_expectation.detach()

        # Compute the loss
        # This is a lower bound of the negative mutual information
        loss = -(torch.mean(mi_estimate) - torch.log(self.ma_et + 1e-6))
        
        return loss
