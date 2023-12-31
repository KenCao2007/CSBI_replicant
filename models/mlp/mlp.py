import torch
import torch.nn as nn
from models.utils import *
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        input_dim = config.model_configs['mlp'].inputdim
        hidden_dim = config.model_configs['mlp'].hiddendim
        output_dim = config.model_configs['mlp'].outputdim

        self.zero_out_last_layer = False
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


