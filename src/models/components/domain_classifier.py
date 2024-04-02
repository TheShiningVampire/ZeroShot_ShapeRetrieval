from torch import nn
import torchvision


class Domain_Classifier(nn.Module):
    def __init__(
        self,
        ):
        super().__init__()

        # Replace the last layer with a new one
        self.domain_classifier = nn.Sequential(
                                nn.Linear(300, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64, 2)
                        ) 

    def forward(self, input):
        return self.domain_classifier(input)

