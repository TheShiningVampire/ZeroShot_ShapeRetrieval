from torch import nn
import torchvision


class Domain_Disentangled_Img_Classifier(nn.Module):
    def __init__(
        self,
        num_classes=20
        ):
        super().__init__()

        # Replace the last layer with a new one
        self.image_classifier = nn.Sequential(
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(256, num_classes),
                                    nn.Softmax(dim=1)
                                )   

    def forward(self, input):
        return self.image_classifier(input)

