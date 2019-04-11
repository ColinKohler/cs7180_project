import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

import utils

###########################################################################################################################################
#                                                         Logical Modules                                                                 #
###########################################################################################################################################

class And(nn.Module):
  def __init__(self):
    super(And, self).__init__()
    self.num_attention_maps = 2
    self.name = 'And'

  def forward(self, attention):
    # Soft-logical and (Min)
    return torch.min(attention, dim=1)[0]

class Or(nn.Module):
  def __init__(self):
    super(Or, self).__init__()
    self.num_attention_maps = 2
    self.name = 'Or'

  def forward(self, attention):
     # Soft-logical or (Max)
    return torch.max(attention, dim=1)[0]

class Id(nn.Module):
  def __init__(self):
    super(Id, self).__init__()
    self.num_attention_maps = 1
    self.name = 'Id'

  def forward(self, input):
    return F.relu(input)


###########################################################################################################################################
#                                                         Complex Modules                                                                 #
###########################################################################################################################################

class Find(nn.Module):
  def __init__(self, context_dim, map_dim=64, text_dim=1):
    super(Find, self).__init__()
    self.num_attention_maps = 0
    self.name = 'Find'

    self.context_dim = context_dim
    self.map_dim = map_dim
    self.kernel_size = 1
    self.text_dim = text_dim

    # conv2(conv1(xvis), W*xtxt)
    self.fc1 = nn.Linear(self.text_dim, self.map_dim)
    self.conv1 = nn.Conv2d(self.context_dim[0], self.map_dim, self.kernel_size)
    self.conv2 = nn.Conv2d(self.map_dim, 1, self.kernel_size)
    self.sigmoid = nn.Sigmoid()

    # Use Xavier init
    utils.xavierInit(self.fc1)
    utils.xavierInit(self.conv1)
    utils.xavierInit(self.conv2)

  def forward(self, context, text):
    batch_size = context.size(0)
    text_mapped = F.relu(self.fc1(text.view(batch_size, -1))).view(batch_size, self.map_dim, 1, 1)
    context_mapped = F.relu(self.conv1(context))
    eltwise_mult = F.normalize(text_mapped * context_mapped, dim=1)
    return self.sigmoid(self.conv2(eltwise_mult))

class Relocate(nn.Module):
  def __init__(self, context_dim, map_dim=64, text_dim=1):
    super(Relocate, self).__init__()
    self.num_attention_maps = 1
    self.name = 'Relocate'

    self.context_dim = context_dim
    self.map_dim = map_dim
    # self.kernel_size = context_dim[1]
    self.kernel_size = 1
    self.text_dim = text_dim

    # conv2(conv1(xvis), W*xtxt)
    self.fc1 = nn.Linear(self.text_dim, self.map_dim)
    self.conv1 = nn.Conv2d(1, self.map_dim, self.kernel_size)
    self.conv2 = nn.Conv2d(self.map_dim, 1, 1)
    self.sigmoid = nn.Sigmoid()

    # Use Xavier init
    utils.xavierInit(self.fc1)
    utils.xavierInit(self.conv1)
    utils.xavierInit(self.conv2)

  def forward(self, attention, text):
    batch_size = attention.shape[0]
    text_mapped = F.relu(self.fc1(text.view(batch_size, -1)).view(batch_size, self.map_dim, 1, 1))
    attention_mapped = F.relu(self.conv1(attention))
    eltwise_mult = F.normalize(text_mapped * attention_mapped, dim=1)
    return self.sigmoid(self.conv2(eltwise_mult))

class Filter(nn.Module):
  def __init__(self, context_dim, map_dim=64, text_dim=1):
    super(Filter, self).__init__()
    self.num_attention_maps = 1
    self.name = 'Filter'
    self.Find = Find(context_dim, map_dim=map_dim, text_dim=text_dim)
    self.And = And()

  def forward(self, attention, context, text):
    # Combine And and Find
    find_atten = self.Find(context, text)
    return self.And(torch.cat((find_atten, attention), dim=1))

###########################################################################################################################################
#                                                          Answer Modules                                                                 #
###########################################################################################################################################

class Exist(nn.Module):
  def __init__(self, input_dim):
    super(Exist, self).__init__()
    self.num_attention_maps = 1
    self.name = 'Exist'

    self.input_dim = input_dim

    # W * vec(a)
    # self.fc1 = nn.Linear(input_dim[-1]**2, 2)
    self.fc1 = nn.Linear(1, 2)

    # Use Xavier init
    utils.xavierInit(self.fc1)

  def forward(self, attention):
    batch_size = attention.size(0)

    attention = attention.reshape(batch_size, -1)
    max = torch.max(attention, dim=1)[0].view(batch_size, 1)
    # min = torch.min(attention, dim=1)[0].view(batch_size, 1)
    # mean = torch.mean(attention, dim=1).view(batch_size, 1)

    return self.fc1(max)
    # return self.fc1(attention.reshape(batch_size, -1))
