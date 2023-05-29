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
                                nn.Linear(2048, 1536),
                                nn.ReLU(),
                                nn.Dropout(0.2),    
                                nn.Linear(1536, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(1024, 512),
                                nn.ReLU(),  
                            )  

        self.domain_invariant = nn.Sequential(
                                nn.Linear(2048, 1536),
                                nn.ReLU(),
                                nn.Dropout(0.2),    
                                nn.Linear(1536, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(1024, 512),
                                nn.ReLU(),  
                            )  

    def forward(self, input):
        return (self.domain_specific(input), self.domain_invariant(input))

