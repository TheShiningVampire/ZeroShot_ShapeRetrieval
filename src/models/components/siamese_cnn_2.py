from torch import nn
import torchvision
import torch

class Siamese_CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        feature_extractor_num_layers: int
        ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )


    def forward_once(self, x):
        x = torch.squeeze(x)
        x = self.fc(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
