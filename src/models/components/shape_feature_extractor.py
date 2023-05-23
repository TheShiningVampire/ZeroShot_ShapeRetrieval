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
        self.model = torchvision.models.resnet50(pretrained=True)

        # Get output before fc layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.Linear1 = nn.Linear(2048, 1024)
        self.ReLU1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(0.2)
        self.Linear2 = nn.Linear(1024, 512)
        self.ReLU2 = nn.ReLU()
        self.Dropout2 = nn.Dropout(0.2)
        self.Linear3 = nn.Linear(512, 79)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        B, M, C, H, W = input.shape
        input = batch_tensor(input, dim=1,squeeze=True)
        input = self.model(input)
        input = input.view(B*M, -1)
        input = self.Linear1(input)
        input = self.ReLU1(input)
        input = self.Dropout1(input)
        input = self.Linear2(input)
        input = self.ReLU2(input)
        input = self.Dropout2(input)
        input = self.Linear3(input)
        # input = self.log_softmax(input).clone()

        input = unbatch_tensor(input, B, dim=1, unsqueeze=True)
        output = torch.max(input, dim=1)[0]

        return output.squeeze()

