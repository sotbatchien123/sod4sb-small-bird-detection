"""Explicit batch-one BN policy, identical normalization in train and inference.

Keeps historical buffers for checkpoint compatibility but never reads/writes them.
No evaluation labels, shared evaluation statistics, or optimizer updates are used.
"""
from types import MethodType
import torch
import torch.nn.functional as F

def _forward(self, value):
    if value.ndim!=4 or value.shape[0]!=1:
        raise ValueError('Per-image BN policy requires NCHW batch size exactly one')
    return F.batch_norm(value,None,None,self.weight,self.bias,True,0.,self.eps)

def enable_per_image_bn(model):
    count=0
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            module.forward=MethodType(_forward,module)
            count+=1
    if not count: raise ValueError('No BatchNorm2d layers found')
    return count
