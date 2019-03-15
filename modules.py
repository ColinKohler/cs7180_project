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

  def forward(self, attention):
    # Soft-logical and (Min)
    return torch.min(attention, dim=1)[0]

class Or(nn.Module):
  def __init__(self):
    super(Or, self).__init__()
    self.num_attention_maps = 2

  def forward(self, attention):
     # Soft-logical or (Max)
    return torch.max(attention, dim=1)[0]

class Id(nn.Module):
  def __init__(self):
    super(Id, self).__init__()
    self.num_attention_maps = 1

  def forward(self, input):
    return input

###########################################################################################################################################
#                                                         Complex Modules                                                                 #
###########################################################################################################################################

class Find(nn.Module):
  def __init__(self, context_size, num_kernels=64, kernel_size=1, text_dim=64):
    super(Find, self).__init__()
    self.num_attention_maps = 0

    self.context_size = context_size
    self.num_kernels = num_kernels
    self.kernel_size = kernel_size
    self.text_dim = text_dim

    # conv2(conv1(xvis), W*xtxt)
    self.fc1 = nn.Linear(self.text_dim, (self.context_size[1] ** 2) * self.num_kernels)
    self.conv1 = nn.Conv2d(self.context_size[0], self.num_kernels, self.kernel_size)
    self.conv2 = nn.Conv2d(self.num_kernels, 1, self.kernel_size)

  def forward(self, context, text):
    batch_size = context.size(0)
    text_mapped = self.fc1(text).view(batch_size, self.context_size[1], self.context_size[2], self.num_kernels)
    context_mapped = F.relu(self.conv1(context)).permute(0, 2, 3, 1)
    return F.relu(self.conv2((text_mapped  * context_mapped).permute(0, 3, 1, 2)))

class Relocate(nn.Module):
  def __init__(self, input_size, num_kernels=64, kernel_size=5, relocate_where_dim=128):
    super(Relocate, self).__init__()
    self.num_attention_maps = 1

    self.input_size = input_size
    self.num_kernels = num_kernels
    self.kernel_size = kernel_size
    self.relocate_where_dim = relocate_where_dim

    # conv2(conv1(xvis) * W1*sum(a * xvis) * W2*xtxt)
    self.fc1 = nn.Linear(self.input_size[0], (self.input_size[1] ** 2) * self.num_kernels)
    self.fc2 = nn.Linear(self.relocate_where_dim, (self.input_size[1] ** 2) * self.num_kernels)
    self.conv1 = nn.Conv2d(self.input_size[0], self.num_kernels, self.kernel_size)
    self.conv2 = nn.Conv2d(self.num_kernels, 1, self.kernel_size)

  # TODO: relocate_where is a bad name
  def forward(self, attention, context, relocate_where):
    conv_xvis = F.relu(self.conv1(context))
    xvis_attend = F.relu(self.fc1(torch.einsum('ijk,ijl->l', attention, context)))
    W2_xtxt = F.relu(self.fc2(relocate_where))
    return F.relu(self.conv2(conv_xvis * xvis_attend * W2_xtxt))


###########################################################################################################################################
#                                                          Answer Modules                                                                 #
###########################################################################################################################################

class Exist(nn.Module):
  def __init__(self, input_size):
    super(Exist, self).__init__()
    self.num_attention_maps = 1

    self.input_size = input_size

    # W * vec(a)
    self.fc1 = nn.Linear(self.input_size[-1]**2, 1)

  def forward(self, attention):
    batch_size = attention.size(0)
    return self.fc1(attention.reshape(batch_size, -1))
