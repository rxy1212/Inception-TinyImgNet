'''
@file: utils.py
@version: v1.0
@date: 2018-01-07
@author: ruanxiaoyi
@brief: Common utils
@remark: {when} {email} {do what}
'''

# import time
from os.path import join as pjoin
import torch
from torch.autograd import Variable

# def localtime():
#     '''
#     Get current time
#     '''
#     return time.strftime('%Y%m%d%H%M%S', time.localtime())


def save(net, name, state_dict=False):
    '''
    Save a network
    '''
    assert isinstance(name, str), 'name must be a string'
    if state_dict:
        torch.save(net.state_dict(), pjoin('./saved_nets_dict', name + '.pkl'))
    else:
        torch.save(net, pjoin('./saved_nets', name + '.pkl'))


def restore(pkl, model_class=None):
    '''
    Restore a network
    '''
    base_path = './saved_nets'
    if model_class != None:
        try:
            model = model_class()
            return model.load_state_dict(torch.load(pjoin(base_path, pkl)))
        except:
            raise ValueError('model_class must match with the model you want to restore')

    else:
        return torch.load(pjoin(base_path, pkl))


def check_accuracy(net, loader):
    '''
    Check the accuracy of val data
    '''
    print('Checking accuracy on validation set')

    num_correct = 0
    num_samples = 0
    # Put the net in test mode (the opposite of net.train(), essentially)
    net.eval()
    for x, y in loader:
        # reference https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/
        x_var = Variable(x, volatile=True)

        scores = net(x_var.cuda())
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = 100.0 * float(num_correct) / num_samples
    print(f'Got {num_correct} / {num_samples} correct ({acc:.2f}%)')
    return acc


# def adjust_learning_rate(optimizer, decay_rate=0.95):
#     '''
#     Use learning rate decay
#     '''
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = param_group['lr'] * decay_rate
