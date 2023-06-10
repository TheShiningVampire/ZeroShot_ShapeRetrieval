from torch import nn
import torchvision
from src.utils.utils import batch_tensor, unbatch_tensor
import torch

class Shape_Feature_Extractor(nn.Module):
    def __init__(
        self,
        model_choice: int = 2,
        num_classes: int = 20,
        ):
        super().__init__()

        # Model to be used is ResNet 50
        self.model = torchvision.models.resnet50(pretrained=True)

        # Get output before fc layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        # self.Linear1 = nn.Linear(2048, 1024)
        # self.ReLU1 = nn.ReLU()
        # self.Dropout1 = nn.Dropout(0.2)
        # self.Linear2 = nn.Linear(1024, 512)
        # self.ReLU2 = nn.ReLU()
        # self.Dropout2 = nn.Dropout(0.2)
        # self.Linear3 = nn.Linear(512, 20)

        if (model_choice == 0):
            self.layers = nn.Sequential(
                            nn.Linear(2048, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(256, num_classes)
                        )
        elif (model_choice == 1):
            self.layers = nn.Sequential(
                            nn.Linear(2048, 2048),  
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(2048, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(256, num_classes)
                        )
        elif (model_choice == 2):
            self.layers = nn.Sequential(
                            nn.Linear(2048, 1536),
                            nn.ReLU(),
                            nn.Dropout(0.2),    
                            nn.Linear(1536, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(1024, 512),
                            nn.ReLU(),  
                            nn.Dropout(0.2),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(256, num_classes)
                        )






    def forward(self, input):
        B, M, C, H, W = input.shape
        input = batch_tensor(input, dim=1,squeeze=True)
        input = self.model(input)
        input = input.view(B*M, -1)
        # input = self.Linear1(input)
        # input = self.ReLU1(input)
        # input = self.Dropout1(input)
        # input = self.Linear2(input)
        # input = self.ReLU2(input)
        # input = self.Dropout2(input)
        # input = self.Linear3(input)
        # # input = self.log_softmax(input).clone()

        input = unbatch_tensor(input, B, dim=1, unsqueeze=True)
        output = torch.max(input, dim=1)[0]
        output = output.squeeze()
        output = self.layers(output)


        return output

