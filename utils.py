import torch
import torch.nn as nn
import torch.nn.functional as F

def xavierInit(layer, bias=0.0):
  nn.init.xavier_uniform_(layer.weight)
  layer.bias.data.fill_(bias)

def sampleGumble(shape, eps=1e-20):
  U = torch.rand(shape).cuda()
  return -torch.log(-torch.log(U + eps) + eps)

def gumbleSoftmaxSample(logits, temp):
  y = logits + sampleGumble(logits.size())
  return F.softmax(y / temp, dim=2)

def gumbleSoftmax(logits, temp, latent_dim, categorical_dim, hard=False):
  y = gumbleSoftmaxSample(logits, temp)

  if not hard:
    return y.view(-1, latent_dim * categorical_dim)

  shape = y.size()
  _, ind = y.max(dim=-1)
  y_hard = torch.zeros_like(y).view(-1, shape[-1])
  y_hard.scatter_(1, ind.view(-1, 1), 1)
  y_hard = y_hard.view(*shape)
  y_hard = (y_hard - y).detach() + y
  return y_hard.view(-1, latent_dim * categorical_dim)
