import torch


def weight_decay(optimizer, wd):
    """
    Custom weight decay operation, not effecting grad values.
    https://www.fast.ai/2018/07/02/adam-weight-decay/
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            current_lr = group['lr']
            param.data = param.data.add(-wd * group['lr'], param.data)
    return optimizer, current_lr