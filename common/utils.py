'''
@file: utils.py
@version: v1.0
@date: 2018-01-18
@author: ruanxiaoyi
@brief: Common utils
@remark: {when} {email} {do what}
'''


from os.path import join as pjoin
import torch

__all__ = ['save', 'restore']

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
