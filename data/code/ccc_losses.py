import numpy as np

from transformers import AutoFeatureExtractor, AutoModel

import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss_fn(pred, labels, attention_mask=None, model=None):
    # Build a frame-level mask for padded batches
    # attention_mask is sample-level; convert lengths -> frame lengths
    if attention_mask is None:
        frame_mask = torch.ones_like(pred, dtype=torch.bool)
    else:
        sample_lengths = attention_mask.sum(dim=1)  # (B,)
        frame_lengths = model.backbone._get_feat_extract_output_lengths(sample_lengths)  # (B,)
        T = pred.size(1)
        frame_mask = torch.arange(T, device=pred.device)[None, :] < frame_lengths[:, None]  # (B,T)

    loss = F.mse_loss(pred[frame_mask], labels[frame_mask])

    return loss

def ccc_value(est, target, eps = 10**-12):
    """ 
    Concordance correlation coefficient metric 
    
    est: (B,T) predicted valuies, B=no of samples in a batch, T=no of time samples in each batch
    target: (B,T) ground truth values

    return: (B,) ccc for each sample in the batch
    
    """
    est_mu = torch.mean(est,axis=1)
    target_mu = torch.mean(target,axis=1)
    
    centered_est = est - est_mu[:,None]
    centered_target = target - target_mu[:,None]

    cov = torch.mean( centered_est * centered_target, axis=1 )
    var_est = torch.mean(centered_est**2,axis=1)
    var_target = torch.mean(centered_target**2,axis=1)

    ccc = (2*cov) / (var_est + var_target + (est_mu - target_mu)**2 + eps)

    return ccc

def ccc_loss(est, target, eps = 10**-12):
    """ 
    Concordance correlation coefficient loss function
    
    est: (B,T) predicted valuies, B=no of samples in a batch, T=no of time samples in each batch
    target: (B,T) ground truth values

    return: scalar loss value
    """
    ccc = ccc_value(est, target, eps)
    loss = 1-torch.mean(ccc)

    return loss
    
    
 
