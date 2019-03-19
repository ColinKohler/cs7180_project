import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, M_dim, x_dim, device):
    super(Decoder, self).__init__()
    self.device = device
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.max_length = 3 # TODO: This should be set to something smart
    self.M_dim = M_dim
    self.x_dim = x_dim
    self.output_dim = M_dim[0] * M_dim[1] + x_dim

    self.embedding = nn.Embedding(self.input_dim, self.hidden_dim)
    self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.max_length)
    self.fc1 = nn.Linear(self.hidden_dim, 128)
    self.fc2 = nn.Linear(128, self.output_dim)

  def forward(self, query, encoder_outputs, debug=False):
    batch_size = query.size(0)
    embedded = self.embedding(query.permute(1,0))
    attn_weights = F.softmax(self.attn(torch.cat((embedded, self.hidden[0]), dim=2)), dim=2)
    attn_applied = torch.bmm(attn_weights.permute(1,0,2), encoder_outputs.permute(1,0,2))

    attn_query = self.attn_combine(torch.cat((embedded, attn_applied.permute(1,0,2)), dim=2))
    out, self.hidden = self.lstm(F.relu(attn_query), self.hidden)
    out = F.relu(self.fc1(out[-1]))
    out = self.fc2(out)

    M_end = self.M_dim[0] * self.M_dim[1]
    M = out[:,:M_end].view(batch_size,self.M_dim[0], self.M_dim[1])
    x = out[:,M_end:].view(batch_size,1, -1)

    if debug: ipdb.set_trace()
    M = F.softmax(M, dim=1)
    return M, x

  def resetHidden(self, batch_size):
    return (torch.zeros(self.max_length, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.max_length, batch_size, self.hidden_dim).to(self.device))
