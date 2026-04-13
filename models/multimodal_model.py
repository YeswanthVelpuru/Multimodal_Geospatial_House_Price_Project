import torch 
import torch.nn as nn
from models.cnn_model import CNNModel
from models.tabular_model import TabularModel

class MultiModalModel(nn.Module):
    def __init__(self, tabular_input_size):
        super(MultiModalModel, self).__init__()

        self.cnn = CNNModel()
        self.tabular = TabularModel(tabular_input_size)

        self.fc = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tabular(tabular)

        combined = torch.cat((img_feat, tab_feat), dim=1)
        return self.fc(combined)