from torch import nn
import torchvision


class Img_Feature_Extractor(nn.Module):
    def __init__(
        self,
        ):
        super().__init__()

        # Model to be used is ResNet 50
        self.model = torchvision.models.resnet50(pretrained=True)

        # Replace the last layer with a new one
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 79),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.model(input)

