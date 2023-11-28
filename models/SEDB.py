import torch
import torch.nn as nn


class Simple_Edge_Detection_Block(nn.Module):
    def __init__(self,in_channals,num_class=1):
        super(Simple_Edge_Detection_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channals, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, num_class, kernel_size=4, stride=2, padding=1)

    def forward(self, x):

        x1 = self.pool(torch.relu(self.conv1(x)))
        x2 = self.pool(torch.relu(self.conv2(x1)))
        x3 = self.pool(torch.relu(self.conv3(x2)))

        x4 = torch.relu(self.deconv1(x3))
        x5 = torch.relu(self.deconv2(x4))
        x6 = self.deconv3(x5)

        x6=torch.sigmoid(x6)

        return x6