from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F

class Tower(nn.Module):
    def __init__(self):
        super(Tower, self).__init__()

        self.conv1 = nn.Conv2d(3, 256, 2, stride=2)
        self.conv2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.res2 = nn.Conv2d(256, 128, 1, stride=1) # Residual connection
        self.conv3 = nn.Conv2d(128, 256, 2, stride=2)
        self.conv4 = nn.Conv2d(256 + 7, 128, 3, stride=1, padding=1)
        self.res4 = nn.Conv2d(256 + 7, 128, 1, stride=1) # Residual connection
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 1, stride=1)

    def forward(self, x, v):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1)) + self.res2(out1)
        out3 = F.relu(self.conv3(out2))

        ve = v.expand(-1, -1, out3.shape[2], out3.shape[3])
        in4 = torch.cat((out3, ve), 1)

        out4 = F.relu(self.conv4(in4)) + self.res4(in4)
        out5 = F.relu(self.conv5(out4))
        out6 = F.relu(self.conv6(out5))
        return out6
