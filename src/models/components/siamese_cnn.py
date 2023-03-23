from torch import nn
import torchvision

# Resnet does this flatten, but when we split the model this does not stay a part of the forward pass anymore. So we need to implement it ourself
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class Siamese_CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        feature_extractor_num_layers: int
        ):
        super().__init__()

        # Use the FC layers of a pretrained network (ResNet-50)
        resnet50 = torchvision.models.resnet50(pretrained=True)

        # Use layers beyond feature_extractor_num_layers as shared layers
        shared_layers = list(resnet50.children())[feature_extractor_num_layers:-1]
        self.shared_layers = nn.Sequential(*shared_layers, Flatten())

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )


    def forward_once(self, x):
        x = self.shared_layers(x)
        x = self.fc(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
