import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

class Decoder(nn.Module):
  def __init__(self, max_length, hidden_dim, M_dim, device, num_layers=1, mt_norm=1, dropout_prob=0.1):
    super(Decoder, self).__init__()
    self.device = device
    self.max_length = max_length
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.M_dim = M_dim
    self.output_dim = M_dim[0] * M_dim[1]
    self.input_dim = self.output_dim
    self.mt_norm = mt_norm

    self.attn = nn.Linear(self.hidden_dim + self.input_dim, self.max_length)
    self.attn_combine = nn.Linear(self.max_length + self.input_dim, self.hidden_dim)

    self.dropout = nn.Dropout(dropout_prob)
    self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers)
    self.fc1 = nn.Linear(self.hidden_dim, 128)
    self.fc2 = nn.Linear(128, self.output_dim)

  def forward(self, prev_M, encoder_outputs, debug=False):
    batch_size = prev_M.size(1)
    prev_M = self.dropout(prev_M)

    attn_weights = F.softmax(self.attn(torch.cat((prev_M, self.hidden[0]), dim=2)), dim=2)
    # TODO: Mask the padded parts out to 0.
    # TODO: does this make any sort of sense whatsoever (it works?)
    attn_applied = F.relu(torch.einsum('lbs,sbh->lbs', attn_weights, encoder_outputs))

    out = self.attn_combine(torch.cat((prev_M, attn_applied), dim=2))
    out, self.hidden = self.lstm(F.relu(out), self.hidden)
    out = F.relu(self.fc1(out[0]))
    out = self.fc2(out)

    M = out.view(batch_size, self.M_dim[0], self.M_dim[1])

    if (self.mt_norm == 1):
      #   Train Loss:0.52322 | Test Loss:0.50811 | Test Acc:0.757:
      M = F.softmax(M, dim=1)
    elif (self.mt_norm == 2):
      #   Train Loss:0.44937 | Test Loss:0.49296 | Test Acc:0.753:
      M = F.softmax(M.view(batch_size,-1), dim=1).view(batch_size, self.M_dim[0], self.M_dim[1])
    elif (self.mt_norm == 3):
      #   Train Loss:0.44780 | Test Loss:0.49603 | Test Acc:0.740:
      M = F.relu(M)
      tots = torch.clamp(torch.cumsum(M,dim=1)[:,-1],min=1).unsqueeze(1)
      M = (M / tots).view(batch_size, self.M_dim[0], self.M_dim[1])

    if debug: ipdb.set_trace()
    return M, attn_weights, torch.zeros((batch_size, 1))

  def resetHidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
