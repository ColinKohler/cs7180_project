import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import matplotlib.pyplot as plt
import ipdb

from visualizer import Visualizer
from modules import And, Or, Id, Find, Relocate, Exist
from encoders import QueryEncoder, ContextEncoder
from decoders import Decoder

class RNMN(nn.Module):
  def __init__(self, query_lang, embed_dim, num_layers, map_dim, lstm_hidden_dim,
                     device, mt_norm=1, comp_length=5, comp_stop_type=1):
    super(RNMN, self).__init__()
    self.device = device
    self.query_dim = query_lang.num_words
    self.embed_dim = embed_dim
    self.num_layers = num_layers
    self.lstm_hidden_dim = lstm_hidden_dim
    self.map_dim = map_dim
    self.text_dim = self.embed_dim
    self.context_dim = [64, 3, 3]

    self.comp_length = comp_length
    self.comp_stop_type = comp_stop_type
    self.query_lang = query_lang
    self.max_query_len = query_lang.max_len

    # Create attention and answer modules
    self.find = Find(self.context_dim, map_dim=self.map_dim, text_dim=self.text_dim)
    self.relocate = Relocate(self.context_dim, map_dim=self.map_dim, text_dim=self.text_dim)
    self.exist = Exist(self.context_dim)

    self.attention_modules = [And(), self.find, self.relocate]
    self.num_att_modules = len(self.attention_modules)
    self.answer_modules = [self.exist]
    [module.to(self.device) for module in self.attention_modules + self.answer_modules]

    # Create query and context encoders
    self.query_encoder = QueryEncoder(self.query_dim, self.embed_dim, self.lstm_hidden_dim, self.device,
                                      num_layers=self.num_layers, max_query_len=self.max_query_len)
    self.context_encoder = ContextEncoder()

    # Create decoder
    self.M_dim = (self.num_att_modules, sum([m.num_attention_maps for m in self.attention_modules + self.answer_modules]))
    self.decoder = Decoder(self.max_query_len, self.lstm_hidden_dim, self.M_dim,
                           self.device, num_layers=self.num_layers, mt_norm=mt_norm)

    # Create visualizer
    self.visualizer = Visualizer(self.query_lang, self.attention_modules, self.answer_modules, self.comp_length)

  def forward(self, query, query_len, context, debug=False, vis=False, i=0):
    batch_size = query.size(0)

    # Encode the query and context
    self.query_encoder.resetHidden(batch_size)
    embedded_query, encoded_query, self.decoder.hidden = self.query_encoder(query, query_len)
    encoded_context = self.context_encoder(context)

    # Loop over timesteps using modules until a threshold is met
    a_t = torch.zeros((batch_size, self.M_dim[1], self.context_dim[1], self.context_dim[2]), device=self.device)
    M_t = torch.zeros((batch_size, self.M_dim[0], self.M_dim[1]), device=self.device)

    stop_mask = torch.zeros((batch_size, self.comp_length), device=self.device)
    outs = torch.zeros((batch_size, self.comp_length, 2), device=self.device)

    for t in range(self.comp_length):
      if debug: ipdb.set_trace()
      M_t, attn, stop_bits = self.decoder(M_t.view(batch_size, 1, self.M_dim[0]*self.M_dim[1]), encoded_query, query_len, debug=debug)
      x_t = torch.bmm(attn, embedded_query)
      b_t, a_tp1, out = self.forward_1t(encoded_context, a_t, M_t, x_t, debug=debug)

      if vis:
        self.visualizer.visualizeTimestep(t, context, query, attn, x_t, a_t, M_t, b_t, out)

      if (self.comp_stop_type == 1):
        stop_mask[:,t] = stop_bits.squeeze(1)
        outs[:,t,:] = out

      a_t = a_tp1

    if (self.comp_stop_type == 1):
      stop_mask = F.softmax(stop_mask, dim=1)
      out = torch.einsum('bt,bti->bi', stop_mask, outs)

    if debug: ipdb.set_trace()
    if vis: self.visualizer.saveGraph(str(i))
    return F.log_softmax(out, dim=1)

  def forward_1t(self, encoded_context, a_t, M_t, x_t, debug=False):
    batch_size = encoded_context.size(0)
    b_t = torch.zeros((batch_size, self.num_att_modules, self.context_dim[1], self.context_dim[2]), device=self.device)

    # Attention map indexs
    num_att_map_inputs = [module.num_attention_maps for module in self.attention_modules + self.answer_modules]
    attention_map_input_index = np.cumsum(num_att_map_inputs)
    attention_map_input_index = np.insert(attention_map_input_index, 0, 0)

    # Run all attention modules saving output
    for i, module in enumerate(self.attention_modules + self.answer_modules):
      attention = a_t[:,np.arange(attention_map_input_index[i],attention_map_input_index[i+1])]
      if type(module) is Id:
        b_t[:,i] = module.forward(attention.squeeze())
      elif type(module) is And:
        b_t[:,i] = module.forward(attention)
      elif type(module) is Or:
        b_t[:,i] = module.forward(attention)
      elif type(module) is Find:
        b_t[:,i] = module.forward(encoded_context, x_t).squeeze(1)
      elif type(module) is Relocate:
        b_t[:,i] = module.forward(attention, x_t).squeeze(1)
      elif type(module) is Exist:
        out = module.forward(attention)
      else:
        raise ValueError('Invalid Module: {}'.format(type(module)))

    if debug: ipdb.set_trace()
    a_tp1 = torch.einsum('bkij,bkl->blij', b_t, M_t)
    return b_t, a_tp1, out

  def saveModel(self, path):
    torch.save(self.state_dict(), path)

  def loadModel(self, path):
    self.load_state_dict(torch.load(path))
