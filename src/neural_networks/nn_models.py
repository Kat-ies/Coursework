"""
classes:
Net
ConvNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_len):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_len, 70)
        self.fc2 = nn.Linear(70, 35)
        self.fc3 = nn.Linear(35, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(in_features=2 * 2 * 64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.layer1(x), 3))
        out = F.relu(F.max_pool2d(self.layer2(out), 2))
        out = self.drop_out(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out
