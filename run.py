'''
@file: run.py
@version: v1.0
@date: 2018-01-18
@author: ruanxiaoyi
@brief: Run the network
@remark: {when} {email} {do what}
'''

import os
from colored import fore, style
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from common.net import InceptionV1
from common.dataset import TIN200Data
from common.utils import *


def train(net, loss_fn, optimizer, num_epochs, epoch, loader=None):
    print(fore.LIGHT_BLUE + f'Starting epoch {epoch} / {num_epochs}' + style.RESET)
    num_correct = 0
    num_samples = 0
    net.train()
    for t, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x_train = Variable(x.cuda())
        y_train = Variable(y.cuda())

        scores, scores0, scores1 = net(x_train)
        scores_all = scores*0.7 + scores0*0.2 + scores1*0.1
        loss = loss_fn(scores_all, y_train)

        loss.backward()
        optimizer.step()
        # reference https://discuss.pytorch.org/t/argmax-with-pytorch/1528
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        acc = 100.0 * float(num_correct) / num_samples
        if (t + 1) % 50 == 0:
            print(
                f't = {t + 1}, loss = {loss.data[0]:.4f}, acc = {acc:.2f}%')


def predict(net, name, loader):
    print('Predicting on test set')
    classid = []
    test_img_name = []
    classid_map = {}

    for _, _, files in os.walk(pjoin('/data1/tiny-imagenet-200', 'test')):
        if files:
            test_img_name = files

    with open(pjoin('/data1/tiny-imagenet-200', 'wnids.txt'), 'r') as f:
        content = [x.strip() for x in f.readlines()]
        classid_map = {index: classid for index,
                       classid in enumerate(content)}

    net.eval()
    for x in loader:
        x_var = Variable(x, volatile=True)
        scores = net(x_var.type(torch.cuda.FloatTensor))
        _, preds = scores.data.cpu().max(1)
        classid += [classid_map[p] for p in preds]

    with open(pjoin(os.getcwd(), 'predictions', f'{name}.txt'), 'w') as f:
        for i in range(len(classid)):
            f.write(f'{test_img_name[i]} {classid[i]}\n')
            if (i + 1) % 500 == 0:
                print(f'process:{i+1}/{len(classid)}')


def main(flag=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
    torch.cuda.is_available()

    train_loader = data.DataLoader(TIN200Data('/data1'), 256, True, num_workers=4)
    val_loader = data.DataLoader(TIN200Data('/data1', 'val'), 256, num_workers=4)

    net = InceptionV1().cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    num_epochs = 300
    for epoch in range(num_epochs):
        train(net, loss_fn, optimizer, num_epochs, epoch+1, train_loader)
        acc = check_accuracy(net, val_loader)
        scheduler.step(acc, epoch=epoch + 1)
        print(f'last best_acc:{best_acc:.2f}%')
        if acc > best_acc:
            best_acc = acc
            print(fore.LIGHT_BLUE +
                  f'Got current best_acc:{best_acc:.2f}%, Saving...' + style.RESET)
            save(net, 'InceptionV1')
        current_lr = optimizer.param_groups[0]['lr']
        print(f'current lr:{current_lr}')
        # adjust_learning_rate(optimizer, decay_rate=0.9)
    print('-------------------------------')
    print(f'{best_acc:.2f}%')
    print('-------------------------------')

if __name__ == '__main__':
    main()
