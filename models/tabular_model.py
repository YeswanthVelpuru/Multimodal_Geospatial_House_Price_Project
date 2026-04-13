import torch.nn as nn

class TabularModel(nn.Module):
    def __init__(self, input_size):
        super(TabularModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.model(x)