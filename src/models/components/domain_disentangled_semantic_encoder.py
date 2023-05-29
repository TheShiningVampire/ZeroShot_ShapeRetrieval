from torch import nn
import torchvision


class Domain_Disentangled_Semantic_Encoder(nn.Module):
    def __init__(
        self,
        num_classes=20
        ):
        super().__init__()

        self.semantic_encoder = nn.Sequential(
                                    nn.Linear(300, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 512),
                                    nn.ReLU()
                                )

    def forward(self, input):
        return self.semantic_encoder(input)

