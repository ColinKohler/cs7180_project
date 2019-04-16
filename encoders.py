import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

import utils

class EncDec(nn.Module):
  def __init__(self, input_dim, embed_dim, device, max_query_len, comp_length, M_dim):
    super(EncDec, self).__init__()
    self.device = device
    self.input_dim = input_dim
    self.embed_dim = embed_dim
    self.max_query_len = max_query_len
    self.comp_length = comp_length
    self.output_dim = comp_length*(M_dim + 2*embed_dim)
    self.M_dim = M_dim

    self.word_embeddings = nn.Embedding(self.input_dim, self.embed_dim, padding_idx=0)
    self.conv1 = nn.Linear(self.embed_dim,10)
    self.lin1 = nn.Linear(10*self.max_query_len, 20*self.max_query_len)
    self.lin2 = nn.Linear(20*self.max_query_len, 20*self.max_query_len)
    self.lin3 = nn.Linear(20*self.max_query_len, self.output_dim)
    
    
  def forward(self, query, query_len, debug=False):
    batch_size = query.shape[0]
    embedded = self.word_embeddings(query)
    #c1 = self.conv1(embedded)
    #rl1 = F.relu(c1)
    rl1 = embedded
    rl1 = rl1.view(batch_size,-1)
    fc1 = self.lin1(rl1.view(batch_size,-1))
    rl2 = F.relu(fc1)
    fc2 = self.lin2(rl2)
    rl3 = F.relu(fc2)
    fc3 = self.lin3(rl3)
    rl4 = fc3
    #rl4 = F.relu(fc3)

    M_end = self.comp_length*(self.M_dim)
    M = rl4[:,:M_end]
    M = M.view(batch_size, self.comp_length, self.M_dim)
    M = F.softmax(M, dim=2)
    M = M.permute(1,0,2)
    
    text_dim = self.comp_length*self.embed_dim
    find_text = F.relu(rl4[:,M_end:M_end+text_dim])
    find_text = find_text.view(batch_size, self.comp_length, self.embed_dim)
    find_text = find_text.permute(1,0,2)
    
    text_dim = self.comp_length*self.embed_dim
    reloc_text = F.relu(rl4[:,M_end+text_dim:])
    reloc_text = reloc_text.view(batch_size, self.comp_length, self.embed_dim)
    reloc_text = reloc_text.permute(1,0,2)
    
    return M, find_text, reloc_text
  

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
    return embedded , output, self.hidden

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
