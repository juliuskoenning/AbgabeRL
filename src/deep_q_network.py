import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # define layers: input -> 64 -> 64 -> output
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        # initialize weights and biases
        self._create_weights()

    def _create_weights(self):
        # initialize weights with Xavier uniform and biases to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # forward pass through the layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
