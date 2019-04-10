import torch
import torch.nn as nn

def xavierInit(layer, bias=0.0):
  nn.init.xavier_uniform_(layer.weight)
  layer.bias.data.fill_(bias)
