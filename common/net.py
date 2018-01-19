'''
@file: net.py
@version: v1.0
@date: 2018-01-18
@author: ruanxiaoyi
@brief: Design the network
@remark: {when} {email} {do what}
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

class InceptionV1(nn.Module):
    def __init__(self, num_classes=200, sub_out=False):
        super(InceptionV1, self).__init__()
        self.sub_out = sub_out
        self.bottom_layer = nn.Sequential(
            BasicConv2d(3, 128, 7, padding=3),
            nn.MaxPool2d(3, 1, 1),
            BasicConv2d(128, 256, 1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.pool1 = nn.MaxPool2d(4, 2, 1)
        self.inception1 = Inception(256, 128, 128, 128, 128)
        self.inception2 = Inception(512, 128, 128, 128, 128)
        self.pool2 = nn.MaxPool2d(4, 2, 1)
        self.inception3 = Inception(512, 160, 160, 160, 160)
        if sub_out:
            self.inception_aux0 = InceptionAux(640, num_classes)
        self.inception4 = Inception(640, 64, 256, 256, 64)
        self.inception5 = Inception(640, 128, 256, 256, 128)
        self.inception6 = Inception(768, 128, 256, 256, 128)
        if sub_out:
            self.inception_aux1 = InceptionAux(768, num_classes)
        self.inception7 = Inception(768, 128, 256, 512, 128)
        self.pool3 = nn.MaxPool2d(4, 2, 1)
        self.inception8 = Inception(1024, 128, 256, 512, 128)
        self.inception9 = Inception(1024, 128, 256, 512, 128)
        self.pool4 = nn.AvgPool2d(8)
        self.fcn = nn.Linear(1024, num_classes, bias=False)

    def forward(self, x):
        x = self.bottom_layer(x)
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.inception3(x)
        if self.sub_out and self.training:
            sub_out0 = self.inception_aux0(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        if self.sub_out and self.training:
            sub_out1 = self.inception_aux1(x)
        x = self.inception7(x)
        x = self.pool3(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fcn(x)
        if self.sub_out and self.training:
            return x, sub_out0, sub_out1
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, convL11_out, conv33_out, conv55_out, convR11_out):
        super(Inception, self).__init__()
        self.convL11 = nn.Conv2d(in_channels, convL11_out, 1)
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, conv33_out, 3, padding=1),
            nn.BatchNorm2d(conv33_out),
            nn.ReLU()
        )
        self.conv55 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, conv55_out, 5, padding=2),
            nn.BatchNorm2d(conv55_out),
            nn.ReLU()
        )
        self.convR11 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, convR11_out, 1),
            nn.BatchNorm2d(convR11_out),
            nn.ReLU()
        )

    def forward(self, x):
        x_L11 = self.convL11(x)
        x_33 = self.conv33(x)
        x_55 = self.conv55(x)
        x_R11 = self.convR11(x)
        outputs = (x_L11, x_33, x_55, x_R11)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.pool = nn.AvgPool2d(5, 3, 2)
        self.conv0 = BasicConv2d(in_channels, 128, 1)
        self.conv1 = BasicConv2d(128, 640, 6)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(640, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 16 x 16 x 640
        x = self.pool(x)
        # 6 x 6 x 640
        x = self.conv0(x)
        # 6 x 6 x 128
        x = self.conv1(x)
        # 1 x 1 x 640
        x = x.view(x.size(0), -1)
        # 640
        x = self.fc(x)
        # 200
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ** kwargs):
        super(BasicConv2d, self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.step(x)


# net = InceptionV1()
# print(net)
# x = torch.randn(1, 3, 64, 64)
# y = net(Variable(x))
# print(y.size())
