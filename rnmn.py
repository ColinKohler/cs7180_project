import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import matplotlib.pyplot as plt
import ipdb

from visualizer import Visualizer
from modules import And, Or, Id, Find, Filter, Relocate, Exist
from encoders import EncDec, ContextEncoder
#from decoders import Decoder

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
    self.text_dim = embed_dim
    self.context_dim = [64, 6, 6]

    self.comp_length = comp_length
    self.comp_stop_type = comp_stop_type
    self.query_lang = query_lang
    self.max_query_len = query_lang.max_len

    # Create attention and answer modules
    self.filter = Filter(self.context_dim, map_dim=self.map_dim, text_dim=self.text_dim)
    self.relocate = Relocate(self.context_dim, map_dim=self.map_dim, text_dim=self.text_dim)
    self.exist = Exist(self.context_dim)

    self.attention_modules = [self.filter, self.relocate]
    self.num_att_modules = len(self.attention_modules)
    self.answer_modules = [self.exist]
    [module.to(self.device) for module in self.attention_modules + self.answer_modules]

    # Create query and context encoders
    self.M_dim = self.num_att_modules
    self.enc_dec = EncDec(self.query_dim, self.embed_dim, self.device, self.max_query_len,
                          self.comp_length, self.M_dim)
    self.context_encoder = ContextEncoder()

    # Create visualizer
    self.visualizer = Visualizer(self.query_lang, self.attention_modules, self.answer_modules, self.comp_length)

    self.output_combine = nn.Linear(self.M_dim, 1)

  def forward(self, query, query_len, context, debug=False, vis=False, i=0):
    batch_size = query.size(0)

    # Encode the query and context
    M, find_text, reloc_text = self.enc_dec(query, query_len)
    encoded_context = self.context_encoder(context)

    # Loop over timesteps using modules until a threshold is met
    a_t = torch.ones((batch_size, 1, self.context_dim[1], self.context_dim[2]), device=self.device)
    M_t = torch.zeros((batch_size, self.M_dim), device=self.device)
    attn = torch.zeros((batch_size, self.max_query_len), device=self.device)
    #M = torch.zeros((self.comp_length, batch_size, self.M_dim), device=self.device)

    #stop_mask = torch.zeros((batch_size, self.comp_length), device=self.device)
    outs = torch.zeros((batch_size, self.comp_length, 2), device=self.device)

    for t in range(self.comp_length):
      #if debug: ipdb.set_trace()
      #M_t, attn, stop_bits = self.decoder(M_t.view(batch_size, 1, self.M_dim),
      #                                    attn.view(batch_size, 1, self.max_query_len),
      #                                    encoded_query, query_len, debug=debug)
      # M[t] = M_t
      #x_t = torch.bmm(attn, embedded_query)
      M_t = M[t]
      b_t, a_tp1, out = self.forward_1t(encoded_context, a_t, M_t, find_text[t], reloc_text[t], debug=debug)

      if vis:
        self.visualizer.visualizeTimestep(t, context, query, attn, find_text[t], a_t, M_t, b_t, out)

      #if (self.comp_stop_type == 1):
      #  stop_mask[:,t] = stop_bits.squeeze(1)
      #  outs[:,t,:] = out

      a_t = a_tp1

    #if (self.comp_stop_type == 1):
    #  stop_mask = F.softmax(stop_mask, dim=1)
    #  out = torch.einsum('bt,bti->bi', stop_mask, outs)

    #if debug: ipdb.set_trace()
    if vis: self.visualizer.saveGraph(str(i))
    M_std = torch.var(M,dim=0)
    M_batch_std = torch.var(M,dim=1)
    return F.log_softmax(out, dim=1), M_std, M_batch_std


  def forward_1t(self, encoded_context, a_t, M_t, find_text, reloc_text, debug=False):
    batch_size = encoded_context.size(0)
    b_t = torch.zeros((batch_size, self.num_att_modules, self.context_dim[1], self.context_dim[2]), device=self.device)

    # Run all attention modules saving output
    for i, module in enumerate(self.attention_modules + self.answer_modules):
      if type(module) is Filter:
        b_t[:,i] = module.forward(a_t, encoded_context, find_text).squeeze(1)
      elif type(module) is Relocate:
        b_t[:,i] = module.forward(a_t, reloc_text).squeeze(1)
      elif type(module) is Exist:
        out = module.forward(a_t)
      else:
        raise ValueError('Invalid Module: {}'.format(type(module)))

    if debug: ipdb.set_trace()
    a_tp1 = F.tanh(torch.einsum('bkij,bk->bij', b_t, M_t).unsqueeze(1))
    return b_t, a_tp1, out

  def saveModel(self, path):
    torch.save(self.state_dict(), path)

  def loadModel(self, path):
    self.load_state_dict(torch.load(path))
