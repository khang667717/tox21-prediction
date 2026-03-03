import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim=2048, output_dim=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.model(x)
