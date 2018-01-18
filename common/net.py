'''
@file: net.py
@version: v1.0
@date: 2018-01-18
@author: ruanxiaoyi
@brief: Design the network
@remark: {when} {email} {do what}
'''

import torch.nn as nn
import torch

class InceptionV1(nn.Module):
    def __init__(self):
        super(InceptionV1, self).__init__()
        self.bottom_layer = nn.Sequential(
            nn.Conv2d(3, 256, 7, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(128, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.inception1 = Inception(128, 128, 128, 128, 128)
        self.inception2 = Inception(512, 128, 128, 128, 128)
        self.inception3 = Inception(256, 160, 160, 160, 160)
        self.inception4 = Inception(640, 64, 256, 256, 64, )

    def forward(self, x):
        pass


class Inception(nn.Module):
    def __init__(self, in_channels, convL11_out, conv33_out, conv55_out, convR11_out, sub_out=0):
        super(Inception, self).__init__()
        self.sub_out = sub_out
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
        if not sub_out:
            self.sub_classifier_1 = nn.Sequential(
                nn.AvgPool2d(5, 3, 2),
                nn.Conv2d(in_channels, sub_out, 1),
                nn.BatchNorm2d(sub_out),
                nn.ReLU()
            )
            self.sub_classifier_2 = nn.Sequential(
                nn.Linear(sub_out, sub_out // 2),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(sub_out // 2, 200)
            )

    def forward(self, x):
        x_L11 = self.convL11(x)
        x_33 = self.conv33(x)
        x_55 = self.conv55(x)
        x_R11 = self.convR11(x)
        outputs = (x_L11, x_33, x_55, x_R11)
        if not self.sub_out:
            sub_out = self.sub_classifier_1(x)
            sub_out = sub_out.view(sub_out.size(0), -1)
            sub_out = self.sub_classifier_2(sub_out)
            return torch.cat(outputs, 1), sub_out
        return torch.cat(outputs, 1)
