import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

class Decoder(nn.Module):
  def __init__(self, max_length, hidden_dim, M_dim, x_dim, device, num_layers=1):
    super(Decoder, self).__init__()
    self.device = device
    self.max_length = max_length
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.M_dim = M_dim
    self.x_dim = self.hidden_dim
    self.output_dim = M_dim[0] * M_dim[1]
    self.input_dim = self.output_dim

    self.attn = nn.Linear(self.hidden_dim + self.input_dim, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_dim + self.input_dim, self.hidden_dim)

    self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers)
    self.fc1 = nn.Linear(self.hidden_dim, 128)
    self.fc2 = nn.Linear(128, self.output_dim)

  def forward(self, init_out, encoder_outputs, debug=False):
    batch_size = init_out.size(1)
    attn_weights = F.softmax(self.attn(torch.cat((init_out, self.hidden[0]), dim=2)), dim=2)
    attn_applied = torch.einsum('lbs,sbh->lbh', attn_weights, encoder_outputs)

    out = self.attn_combine(torch.cat((init_out, attn_applied), dim=2))
    out, self.hidden = self.lstm(F.relu(out), self.hidden)
    out = F.relu(self.fc1(out[0]))
    out = self.fc2(out)

    M = out.view(batch_size, self.M_dim[0], self.M_dim[1])
    # x = out[:,M_end:].view(batch_size,1, -1)

    if debug: ipdb.set_trace()
    M = F.softmax(M, dim=1)
    return M, attn_applied

  def resetHidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
