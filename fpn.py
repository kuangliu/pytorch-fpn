'''FPN101(Feature Pyramid Networks) is modified from ResNet101.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top-down layers, add BN?
        self.uplayer1 = nn.ConvTranspose2d(512*4, 256, kernel_size=4, stride=2, padding=1)
        self.uplayer2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.uplayer3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.uplayer4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256*4, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128*4, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 64*4, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        # Bottom-up
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = c5  # TODO: predict here
        print(p5.size())
        p4 = self.uplayer1(p5) + self.latlayer1(c4)  # predict here
        print(p4.size())
        p3 = self.uplayer2(p4) + self.latlayer2(c3)  # predict here
        print(p3.size())
        p2 = self.uplayer3(p3) + self.latlayer3(c2)  # predict here
        print(p2.size())

        return p2


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN101()
    y = net(Variable(torch.randn(1,3,224,224)))
    print(y.size())

test()
