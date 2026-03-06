import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x