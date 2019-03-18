import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import ipdb

from modules import And, Or, Id, Find, Relocate, Exist
from encoders import QueryEncoder, ContextEncoder
from decoders import Decoder

class RNMN(nn.Module):
  def __init__(self, query_size, hidden_size, device):
    super(RNMN, self).__init__()
    self.device = device
    self.query_size = query_size
    self.hidden_size = hidden_size
    self.context_size = [64, 6, 6]

    # Create attention and answer modules
    self.find = Find(self.context_size)
    #self.relocate = Relocate(self.context_size)
    self.exist = Exist(self.context_size)

    self.attention_modules = [Or(), self.find]
    self.num_att_modules = len(self.attention_modules)
    self.answer_modules = [self.exist]
    [module.to(self.device) for module in self.attention_modules + self.answer_modules]

    # Create query and context encoders
    self.query_encoder = QueryEncoder(self.query_size, self.hidden_size, self.device)
    self.context_encoder = ContextEncoder()

    # Create decoder
    self.M_size = (self.num_att_modules, sum([m.num_attention_maps for m in self.attention_modules + self.answer_modules]))
    self.x_size = 64
    self.decoder = Decoder(self.hidden_size, self.M_size, self.x_size, self.device)

  def forward(self, query, query_len, context, debug=False):
    batch_size = query.size(0)

    # Encode the query and context
    encoded_query, query_hidden = self._encodeQuery(query, query_len, debug=debug)
    encoded_context = self.context_encoder(context)

    # Loop over timesteps using modules until a threshold is met
    self.decoder.hidden = query_hidden
    self.a_t = torch.zeros((batch_size, self.M_size[1], self.context_size[1], self.context_size[2]), device=self.device)
    # TODO: This for loop should be replaced with some sort of thresholding junk
    if debug: ipdb.set_trace()
    for t in range(2):
      self.M_t, self.x_t = self.decoder()
      self.a_t, out = self.forward_1t(encoded_context, debug=debug)

    return F.log_softmax(out, dim=1)

  def forward_1t(self, encoded_context, debug=False):
    batch_size = encoded_context.size(0)
    b_t = torch.zeros((batch_size, self.num_att_modules, self.context_size[1], self.context_size[2]), device=self.device)

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
        b_t[:,i] = module.forward(encoded_context, self.x_t).squeeze()
      elif type(module) is Relocate:
        b_t[:,i] = module.forward(attention, encoded_context, self.x_t)
      elif type(module) is Exist:
        out = module.forward(attention.squeeze())
      else:
        raise ValueError('Invalid Module: {}'.format(type(module)))

    if debug: ipdb.set_trace()
    a_tp1 = torch.einsum('bkij,bkl->blij', b_t, self.M_t)
    return a_tp1, out

  def _encodeQuery(self, query, query_len, debug=False):
    batch_size = query.size(0)
    max_query_len = query.size(1)
    query_end_inds = query_len - 1

    encoder_outputs = torch.zeros(batch_size, max_query_len, self.query_encoder.hidden_size, device=self.device)
    encoder_hiddens = torch.zeros(batch_size, max_query_len, self.query_encoder.hidden_size, device=self.device)
    encoder_cell_states = torch.zeros(batch_size, max_query_len, self.query_encoder.hidden_size, device=self.device)

    # Encode each word in the query
    self.query_encoder.resetHidden(batch_size)
    if debug: ipdb.set_trace()
    for ei in range(max_query_len):
      encoder_output, encoder_hidden = self.query_encoder(query[:,ei])
      encoder_outputs[:,ei,:] = encoder_output
      encoder_hiddens[:,ei,:] = encoder_hidden[0]
      encoder_cell_states[:,ei,:] = encoder_hidden[1]

    #Decoder initial state set to hidden state at end of each query
    encoder_hiddens_at_end = encoder_hiddens.gather(1, query_end_inds.view(-1,1).unsqueeze(2).repeat(1, 1, self.hidden_size)).permute(1,0,2)
    encoder_cell_states_at_end = encoder_cell_states.gather(1, query_end_inds.view(-1,1).unsqueeze(2).repeat(1, 1, self.hidden_size)).permute(1,0,2)

    return encoder_outputs, (encoder_hiddens_at_end, encoder_cell_states_at_end)
