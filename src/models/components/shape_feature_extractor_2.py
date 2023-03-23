from torch import nn
import torchvision
from src.utils.utils import batch_tensor, unbatch_tensor
import torch

class Shape_Feature_Extractor(nn.Module):
    def __init__(
        self,
        ):
        super().__init__()

        # Model to be used is ResNet 50
        model = torchvision.models.resnet50(pretrained=True)

        # Get output before fc layer
        self.model = nn.Sequential(*list(model.children())[:-1])
        # self.Linear1 = nn.Linear(2048, 1024)
        # self.ReLU1 = nn.ReLU()
        # self.Dropout1 = nn.Dropout(0.2)
        # self.Linear2 = nn.Linear(1024, 512)
        # self.ReLU2 = nn.ReLU()
        # self.Dropout2 = nn.Dropout(0.2)
        # self.Linear3 = nn.Linear(512, 79)
        # self.log_softmax = nn.LogSoftmax(dim=1)

        # Normalize the output
        self.model = nn.Sequential(
            self.model,
            nn.Flatten(),
            nn.BatchNorm1d(2048),
        )

    def forward(self, input):
        input = self.model(input)


        return input.squeeze()

