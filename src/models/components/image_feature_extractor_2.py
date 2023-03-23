from torch import nn
import torchvision


class Img_Feature_Extractor(nn.Module):
    def __init__(
        self,
        ):
        super().__init__()

        # Model to be used is ResNet 50
        model = torchvision.models.resnet50(pretrained=True)

        # Remove the last layer
        self.model = nn.Sequential(*list(model.children())[:-1])
        
        # Normalize the output
        self.model = nn.Sequential(
            self.model,
            nn.Flatten(),
            nn.BatchNorm1d(2048),
        )

    def forward(self, input):
        return self.model(input)

