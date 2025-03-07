# models.py

import torch.nn as nn

class ImitationNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImitationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)
