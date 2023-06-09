from torch import nn
import torchvision


class Domain_Disentangled_Shape_Classifier(nn.Module):
    def __init__(
        self,
        num_classes=20
        ):
        super().__init__()

        # Replace the last layer with a new one
        self.image_classifier = nn.Sequential(
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, num_classes)
                        ) 

    def forward(self, input):
        return self.image_classifier(input)

