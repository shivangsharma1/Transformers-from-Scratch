import torch
import torch.nn as nn

class VanillaNN(nn.Module):
    def __init__(self, attention_dim):
        super().__init__()
        torch.manual_seed(0)

        self.up_projection = nn.Linear(attention_dim, attention_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(attention_dim*4, attention_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.up_projection(x)
        x = self.relu(x)
        x = self.down_projection(x)
        x = self.dropout(x)

        return x