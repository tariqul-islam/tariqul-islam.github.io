import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def phi(x,y,a,b):
    '''Kernel Function'''
    factor = 1 + a * torch.sum((x-y)**2) ** b
    y = 1/factor
    
    return y

def criterion_pos(x,y,a,b):
    '''Positive Loss'''
    prob = phi(x,y,a,b)
    y = -torch.log(prob)
    
    return y

def criterion_neg(x,y,a,b):
    '''Negative Loss'''
    prob = phi(x,y,a,b)
    y = -torch.log(1-prob+0.0001)
    
    return y
