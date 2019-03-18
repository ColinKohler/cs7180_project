import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
  def __init__(self, hidden_dim, M_dim, x_dim, device, num_layers=1):
    super(Decoder, self).__init__()
    self.device = device
    self.hidden_dim = hidden_dim
    self.M_dim = M_dim
    self.x_dim = x_dim
    self.num_layers = num_layers
    self.output_dim = M_dim[0] * M_dim[1] + x_dim
    self.hidden = self.resetHidden(1)

    self.lstm = nn.LSTM(hidden_dim, hidden_dim)
    self.fc1 = nn.Linear(hidden_dim, 128)
    self.fc2 = nn.Linear(128, self.output_dim)

  def forward(self):
    # TODO: LSTMs have to have input but I dunno what it would be here.  (Currently Zeros)
    out, self.hidden = self.lstm(torch.zeros(self.hidden[0].shape, device=self.device), self.hidden)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)

    batch_size = out.shape[1]

    M_end = self.M_dim[0] * self.M_dim[1]
    M = out[:,:,:M_end].view(batch_size,self.M_dim[0], self.M_dim[1])
    x = out[:,:,M_end:].view(batch_size,1, -1)

    M = F.softmax(M, dim=2)
    return M, x

  def resetHidden(self, batch_size):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
