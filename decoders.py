import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import ipdb

import utils

class Decoder(nn.Module):
  def __init__(self, max_length, hidden_dim, M_dim, device, num_layers=1, mt_norm=1, dropout_prob=0.1):
    super(Decoder, self).__init__()
    self.device = device
    self.max_length = max_length
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.M_dim = M_dim
    self.output_dim = M_dim
    self.input_dim = self.output_dim
    self.mt_norm = mt_norm

    self.attn = Attention(self.hidden_dim, self.max_length)

    # self.composition_expand = nn.Linear(self.output_dim, self.hidden_dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.lstm = nn.LSTM(self.output_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.5)
    self.decode_head = nn.Linear(self.hidden_dim, self.output_dim)

    # Use Xavier init
    utils.xavierInit(self.decode_head)

  def forward(self, prev_M, encoder_outputs, query_len, debug=False):
    batch_size = prev_M.size(0)
    prev_M = self.dropout(prev_M)

    # prev_M = self.composition_expand(prev_M)
    out, self.hidden = self.lstm(prev_M, self.hidden)
    mask = torch.arange(self.max_length, device=self.device).expand(len(query_len), self.max_length) >= query_len.unsqueeze(1)
    out, attn_weights = self.attn(out, encoder_outputs, mask=mask.unsqueeze(1), temp=1.0)

    out = self.decode_head(out.view(batch_size, -1))
    M = out.view(batch_size, self.M_dim)

    if (self.mt_norm == 1):
      M = F.softmax(M, dim=1)
    elif (self.mt_norm == 2):
      M = F.softmax(M.view(batch_size,-1), dim=1).view(batch_size, self.M_dim)
    elif (self.mt_norm == 3):
      M = F.relu(M)
      tots = torch.clamp(torch.cumsum(M,dim=1)[:,-1],min=1).unsqueeze(1)
      M = (M / tots).view(batch_size, self.M_dim)

    if debug: ipdb.set_trace()
    return M, attn_weights, torch.zeros((batch_size, 1))

  def resetHidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))

class Attention(nn.Module):
  def __init__(self, hidden_dim, max_length):
    super(Attention, self).__init__()

    self.hidden_dim = hidden_dim
    self.max_length = max_length

    self.linear_out = nn.Linear(self.hidden_dim*2, self.hidden_dim)
    self.tanh = nn.Tanh()

    # Use Xavier init
    utils.xavierInit(self.linear_out)

  def forward(self, output, context, mask=None, temp=1.0):
    batch_size = output.size(0)
    hidden_size = output.size(2)
    input_size = context.size(1)

    # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
    attn = torch.bmm(output, context.permute(0,2,1))
    if not mask is None:
      attn.data.masked_fill_(mask, -float('inf'))
    attn = F.softmax(attn.view(-1, input_size)/temp, dim=1).view(batch_size, -1, input_size)

    # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
    mix = torch.bmm(attn, context)

    # concat -> (batch, out_len, 2*dim)
    combined = torch.cat((mix, output), dim=2)
    # output -> (batch, out_len, dim)
    output = self.tanh(self.linear_out(combined.view(-1, 2* hidden_size))).view(batch_size, -1, hidden_size)

    return output, attn
