import torch 
import torch.nn as nn
from torchsummary import summary
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        n_features = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = nn.Sequential(self.conv1, self.conv2, self.conv3)(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # CONVOLUTIONAL
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = x.flatten(1)

        # FULLY CONNECTED
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

#net = DQN((1,84,84), 5)
#summary(net, (1,84,84))
