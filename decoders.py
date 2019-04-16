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
    self.input_dim = self.output_dim # + self.max_length + 4
    self.mt_norm = mt_norm
    self.temp = 5.0
    self.temp_min = 0.5
    self.temp_anneal_rate = 0.00075

    self.attn = Attention(self.input_dim, self.hidden_dim, self.max_length)

    # self.composition_expand = nn.Linear(self.output_dim, self.hidden_dim)
    self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.5)
    self.decode_head = nn.Linear(self.hidden_dim, self.output_dim)

    # Use Xavier init
    utils.xavierInit(self.decode_head)

  def forward(self, prev_M, prev_attn, inp_attn, encoder_outputs, query_len, debug=False):
    batch_size = prev_M.size(0)
    # prev_data = self.dropout(torch.cat((prev_M, prev_attn, inp_attn), dim=-1))
    prev_data = prev_M

    mask = torch.arange(self.max_length, device=self.device).expand(len(query_len), self.max_length) >= query_len.unsqueeze(1)
    out, attn_weights = self.attn(prev_data, self.hidden[0][-1].view(1, batch_size, self.hidden_dim), encoder_outputs, mask=mask.unsqueeze(1), temp=1.0)

    out, self.hidden = self.lstm(out, self.hidden)

    out = self.decode_head(out.view(batch_size, -1))
    M = out.view(batch_size, self.M_dim)

    if (self.mt_norm == 1):
      M = F.gumbel_softmax(M, self.temp, hard=True)
      # M = F.softmax(M, dim=1)
      # M = F.sigmoid(M)
    elif (self.mt_norm == 2):
      M = F.softmax(M.view(batch_size,-1), dim=1).view(batch_size, self.M_dim)
    elif (self.mt_norm == 3):
      M = F.relu(M)
      tots = torch.clamp(torch.cumsum(M,dim=1)[:,-1],min=1).unsqueeze(1)
      M = (M / tots).view(batch_size, self.M_dim)

    if debug: ipdb.set_trace()
    return M, attn_weights, torch.zeros((batch_size, 1))

  def annealTemp(self, epoch):
    self.temp = np.maximum(self.temp*np.exp(-self.temp_anneal_rate*epoch), self.temp_min)

  def resetHidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))

class Attention(nn.Module):
  def __init__(self, input_dim, hidden_dim, max_length, dropout_prob=0.5):
    super(Attention, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.max_length = max_length

    self.expand_input = nn.Linear(self.input_dim, self.hidden_dim)
    self.attn = nn.Linear(self.hidden_dim*2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim)
    self.dropout = nn.Dropout(dropout_prob)

    # Use Xavier init
    utils.xavierInit(self.expand_input)
    utils.xavierInit(self.attn)
    utils.xavierInit(self.attn_combine)

  def forward(self, inp, hidden, context, mask=None, temp=1.0):
    batch_size = inp.size(0)
    input_size = context.size(1)

    inp = self.expand_input(inp)
    inp = self.dropout(inp)
    attn_weights = self.attn(torch.cat((inp, hidden.permute(1,0,2)), dim=2))
    if not mask is None:
      attn_weights.data.masked_fill_(mask, -float('inf'))
    attn_weights = F.softmax(attn_weights.view(batch_size, input_size)/temp, dim=-1).view(batch_size, -1, input_size)

    # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
    attn_applied = torch.bmm(attn_weights, context)

    # concat -> (batch, out_len, 2*dim)
    combined = torch.cat((inp, attn_applied), dim=2)
    # output -> (batch, out_len, dim)
    output = self.attn_combine(combined.view(batch_size, -1)).view(batch_size, -1, self.hidden_dim)

    return output, attn_weights
