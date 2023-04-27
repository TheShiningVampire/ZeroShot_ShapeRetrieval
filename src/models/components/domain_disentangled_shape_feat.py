from torch import nn
import torchvision


class Domain_Disentangled_Shape_Feature(nn.Module):
    def __init__(
        self,
        num_classes=20
        ):
        super().__init__()

        # Replace the last layer with a new one
        self.domain_specific = nn.Sequential(
                                    nn.Linear(2048, 2048),
                                    nn.Sigmoid(),
                                    nn.BatchNorm1d(2048),
                                    nn.Linear(2048, 1024),
                                    nn.Sigmoid(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 1024),
                                    nn.Sigmoid(),
                                    nn.BatchNorm1d(1024)
                                )   

        self.domain_invariant = nn.Sequential(
                                    nn.Linear(2048, 2048),
                                    nn.Sigmoid(),
                                    nn.BatchNorm1d(2048),
                                    nn.Linear(2048, 1024),
                                    nn.Sigmoid(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 1024),
                                    nn.Sigmoid(),
                                    nn.BatchNorm1d(1024)
                                )

    def forward(self, input):
        return (self.domain_specific(input), self.domain_invariant(input))

