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
  def __init__(self, query_lang, embed_dim, num_layers, map_dim, lstm_hidden_dim, text_dim, device, mt_norm=1, comp_length=5, comp_stop_type=1):
    super(RNMN, self).__init__()
    self.device = device
    self.query_dim = query_lang.num_words
    self.embed_dim = embed_dim
    self.lstm_hidden_dim = lstm_hidden_dim
    self.text_dim = text_dim
    self.num_layers = num_layers
    self.map_dim = map_dim
    self.context_dim = [64, 6, 6]
    self.comp_length = comp_length
    self.comp_stop_type = comp_stop_type
    self.max_query_len = query_lang.max_len
    self.query_lang = query_lang

    # Create attention and answer modules
    self.find = Find(self.context_dim, map_dim=self.map_dim, text_dim=self.text_dim)
    self.relocate = Relocate(self.context_dim, map_dim=self.map_dim, text_dim=self.text_dim)
    self.exist = Exist(self.context_dim)

    self.attention_modules = [And(), self.find]# , self.relocate]
    self.num_att_modules = len(self.attention_modules)
    self.answer_modules = [self.exist]
    [module.to(self.device) for module in self.attention_modules + self.answer_modules]

    # Create query and context encoders
    self.query_encoder = QueryEncoder(self.query_dim, self.embed_dim, self.lstm_hidden_dim, self.device, num_layers=self.num_layers, max_query_len=self.max_query_len)
    self.query_combine = nn.Linear(self.max_query_len*self.embed_dim, self.text_dim)
    self.context_encoder = ContextEncoder()

    # Create decoder
    self.M_dim = (self.num_att_modules, sum([m.num_attention_maps for m in self.attention_modules + self.answer_modules]))
    self.decoder = Decoder(self.max_query_len, self.lstm_hidden_dim, self.M_dim, self.device, num_layers=self.num_layers, mt_norm=mt_norm)

    # Create visualizer
    self.visualizer = Visualizer(self.query_lang, self.attention_modules, self.answer_modules, self.comp_length)

  def forward(self, query, query_len, context, debug=False, vis=False):
    batch_size = query.size(0)

    # Encode the query and context
    embedded_query, encoded_query, query_hidden = self._encodeQuery(query, query_len, debug=debug)
    encoded_context = self.context_encoder(context)

    # Loop over timesteps using modules until a threshold is met
    self.decoder.hidden = query_hidden
    self.a_t = torch.zeros((batch_size, self.M_dim[1], self.context_dim[1], self.context_dim[2]), requires_grad=True, device=self.device)
    self.M_t = torch.zeros((batch_size, self.M_dim[0], self.M_dim[1]), requires_grad=True, device=self.device)

    self.stop_mask = torch.zeros((batch_size, self.comp_length), device=self.device)
    self.outs = torch.zeros((batch_size, self.comp_length, 2), device=self.device)

    for t in range(self.comp_length):
      if debug: ipdb.set_trace()
      self.M_t, attn, stop_bits = self.decoder(self.M_t.view(batch_size, self.M_dim[0]*self.M_dim[1], 1).permute(2,0,1), encoded_query, debug=debug)
      self.x_t = self.query_combine(torch.einsum('dbe,ebn->ben', attn, embedded_query).view(batch_size, -1))
      b_t, a_tp1, out = self.forward_1t(encoded_context, debug=debug)

      if vis:
        self.visualizer.visualizeTimestep(t, context, query, self.a_t, self.M_t, b_t, out)

      if (self.comp_stop_type == 1):
        self.stop_mask[:,t] = stop_bits.squeeze(1)
        self.outs[:,t,:] = out

      self.a_t = copy.copy(a_tp1)

    if (self.comp_stop_type == 1):
      self.stop_mask = F.softmax(self.stop_mask,dim = 1)
      out = torch.einsum('bt,bti->bi',self.stop_mask,self.outs)

    if debug: ipdb.set_trace()
    if vis: self.visualizer.saveGraph()
    return F.log_softmax(out, dim=1)

  def forward_1t(self, encoded_context, debug=False):
    batch_size = encoded_context.size(0)
    b_t = torch.zeros((batch_size, self.num_att_modules, self.context_dim[1], self.context_dim[2]), device=self.device)

    # Attention map indexs
    num_att_map_inputs = [module.num_attention_maps for module in self.attention_modules + self.answer_modules]
    attention_map_input_index = np.cumsum(num_att_map_inputs)
    attention_map_input_index = np.insert(attention_map_input_index, 0, 0)

    # Run all attention modules saving output
    for i, module in enumerate(self.attention_modules + self.answer_modules):
      attention = self.a_t[:,np.arange(attention_map_input_index[i],attention_map_input_index[i+1])]
      if type(module) is Id:
        b_t[:,i] = module.forward(attention.squeeze())
      elif type(module) is And:
        b_t[:,i] = module.forward(attention)
      elif type(module) is Or:
        b_t[:,i] = module.forward(attention)
      elif type(module) is Find:
        b_t[:,i] = module.forward(encoded_context, self.x_t).squeeze(1)
      elif type(module) is Relocate:
        b_t[:,i] = module.forward(attention, encoded_context, self.x_t).squeeze(1)
      elif type(module) is Exist:
        out = module.forward(attention)
      else:
        raise ValueError('Invalid Module: {}'.format(type(module)))

    if debug: ipdb.set_trace()
    a_tp1 = torch.einsum('bkij,bkl->blij', b_t, self.M_t)
    return b_t, a_tp1, out

  def _encodeQuery(self, query, query_len, debug=False):
    batch_size = query.size(0)
    max_query_len = query.size(1)
    query_end_inds = query_len - 1

    # Encode each word in the query
    self.query_encoder.resetHidden(batch_size)
    if debug: ipdb.set_trace()
    embeded_query, outputs, (hidden_states, cell_states) = self.query_encoder(query, query_len)

    return embeded_query, outputs, (hidden_states, cell_states)

  def saveModel(self, save_path):
    pass

  def loadModel(self, load_path):
    pass
