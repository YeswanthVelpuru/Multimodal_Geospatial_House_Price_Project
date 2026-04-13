import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Load pretrained ResNet18
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # New fully connected layer
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x