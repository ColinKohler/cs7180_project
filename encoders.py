import torch
import torch.nn as nn
import torch.nn.functional as F


import ipdb

class QueryEncoder(nn.Module):
  def __init__(self, input_size, embed_size, hidden_size, device, num_layers=1, max_query_len=7):
    super(QueryEncoder, self).__init__()
    self.device = device
    self.input_size = input_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.max_query_len = max_query_len

    self.word_embeddings = nn.Embedding(self.input_size, self.embed_size)
    self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers)

  def resetHidden(self, batch_size):
    self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                   torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))

  def forward(self, query, query_len):
    #ipdb.set_trace()
    batch_size = query.size(0)
    embedded = self.word_embeddings(query.permute(1,0))
    packed = nn.utils.rnn.pack_padded_sequence(embedded, query_len)
    lstm_out, self.hidden = self.lstm(packed, self.hidden)
    output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, total_length=self.max_query_len)
    return embedded, output, self.hidden

class ContextEncoder(nn.Module):
  def __init__(self):
    super(ContextEncoder, self).__init__()

    # Init two conv layers to extract features (64 kernels)
    self.conv1 = nn.Conv2d(3, 64, 10, stride=10)
    self.conv2 = nn.Conv2d(64, 64, 1, stride=1)

  def forward(self, context):
    return F.relu(self.conv2(F.relu(self.conv1(context))))
