import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

import utils

class QueryEncoder(nn.Module):
  def __init__(self, input_dim, embed_dim, hidden_dim, device, num_layers=1, max_query_len=7, dropout_prob=0.1):
    super(QueryEncoder, self).__init__()
    self.device = device
    self.input_dim = input_dim
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.max_query_len = max_query_len

    self.word_embeddings = nn.Embedding(self.input_dim, self.embed_dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.5, bidirectional=True)

  def resetHidden(self, batch_dim):
    self.hidden = (torch.zeros(self.num_layers*2, batch_dim, self.hidden_dim).to(self.device),
                   torch.zeros(self.num_layers*2, batch_dim, self.hidden_dim).to(self.device))

  def forward(self, query, query_len, debug=False):
    if debug: ipdb.set_trace()
    embedded = self.word_embeddings(query)
    embedded = self.dropout(embedded)

    packed = nn.utils.rnn.pack_padded_sequence(embedded, query_len, batch_first=True)
    lstm_out, self.hidden = self.lstm(packed, self.hidden)
    output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, total_length=self.max_query_len, batch_first=True)
    output = output[:,:,:self.hidden_dim] + output[:,:,self.hidden_dim:]
    return embedded, output, self.hidden

class ContextEncoder(nn.Module):
  def __init__(self):
    super(ContextEncoder, self).__init__()

    # Init two conv layers to extract features (64 kernels)
    self.conv1 = nn.Conv2d(3, 64, 10, stride=10)
    self.conv2 = nn.Conv2d(64, 64, 1, stride=1)

    # Use Xavier init
    utils.xavierInit(self.conv1)
    utils.xavierInit(self.conv2)

  def forward(self, context):
    return F.relu(self.conv2(F.relu(self.conv1(context))))
