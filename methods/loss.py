from typing import Union

import torch
import torch.nn.functional as F

import geoopt


# alignment loss (https://github.com/ssnl/align_uniform)
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

# uniform loss  (https://github.com/ssnl/align_uniform)
def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

# DSR loss
def subspace_loss(x, subspace, detach=True, norm=True):
    '''
    x: (N * D)
    subspace: (D * k)
    '''
    projection = x @ subspace @ subspace.T
    if detach:
        projection = projection.detach()
    if norm:
        projection = F.normalize(projection, dim=1)

    loss = torch.norm(x - projection, p='fro') / x.shape[0]
    return loss

# PD loss
def distillation_loss(z_enc, z_pro, temperature=0.5):
    logits_enc = F.log_softmax(z_enc / temperature, dim=1)
    logits_pro = F.softmax(z_pro / temperature, dim=1)
    return F.kl_div(logits_enc, logits_pro, reduction='batchmean')

def mse_loss(z_enc, z_pro):
    return torch.mean((z_enc - z_pro) ** 2)

