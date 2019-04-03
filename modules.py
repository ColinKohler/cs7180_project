import torch
import torch.nn as nn
import torch.nn.functional as F

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
  def __init__(self, context_dim, map_dim=64, text_dim=256):
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

  def forward(self, context, text):
    batch_size = context.size(0)
    text_mapped = F.relu(self.fc1(text.view(batch_size, -1))).view(batch_size, self.map_dim, 1, 1)
    context_mapped = F.relu(self.conv1(context))
    eltwise_mult = F.normalize(torch.einsum('bmxx,bmhw->bmhw', text_mapped, context_mapped), dim=1)
    return self.sigmoid(self.conv2(eltwise_mult))

class Relocate(nn.Module):
  def __init__(self, context_dim, map_dim=64, text_dim=256):
    super(Relocate, self).__init__()
    self.num_attention_maps = 1
    self.name = 'Relocate'

    self.context_dim = context_dim
    self.map_dim = map_dim
    self.kernel_size = 1
    self.text_dim = text_dim

    # conv2(conv1(xvis) * W1*sum(a * xvis) * W2*xtxt)
    self.fc1 = nn.Linear(self.text_dim, self.map_dim)
    self.fc2 = nn.Linear(self.context_dim[0], self.map_dim)
    self.conv1 = nn.Conv2d(self.context_dim[0], self.map_dim, 1)
    self.conv2 = nn.Conv2d(self.map_dim, 1, 1)

  def forward(self, attention, context, text):
    batch_size = attention.shape[0]
    text_mapped = F.relu(self.fc1(text.view(batch_size, -1)).view(batch_size, self.map_dim, 1, 1))
    context_mapped = F.relu(self.conv1(context))

    attention_softmax = F.softmax(attention.view(batch_size, -1), dim=1)
    attention_softmax = attention_softmax.view(batch_size, 1, self.context_dim[1], self.context_dim[2])
    attention = torch.sum(context * attention_softmax, dim=[2,3])
    attention_mapped = F.sigmoid(self.fc2(attention)).view(batch_size, self.map_dim, 1, 1)

    eltwise_mult = F.normalize(context_mapped * text_mapped * attention_mapped)
    return F.sigmoid(self.conv2(eltwise_mult))

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
    self.fc1 = nn.Linear(input_dim[-1]**2, 2)

  def forward(self, attention):
    batch_size = attention.size(0)
    return self.fc1(attention.reshape(batch_size, -1))
