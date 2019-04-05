import shutil
import os, sys
import pygraphviz as pgv
import matplotlib.pyplot as plt

class Visualizer(object):
  def __init__(self, query_lang, attention_modules, answer_modules, comp_length):
    self.query_lang = query_lang
    self.attention_modules = attention_modules
    self.answer_modules = answer_modules
    self.modules = self.attention_modules + self.answer_modules
    self.comp_length = comp_length

    self.module_output_names = [m.name for m in self.attention_modules]
    self.module_input_names = list()
    for module in self.modules:
      self.module_input_names.extend([module.name] * module.num_attention_maps)

    self.G = pgv.AGraph(directed=True)
    self.G.node_attr['shape'] = 'box'
    self.G.node_attr['labelloc'] = 't'
    self.G.node_attr['margin'] = '0.22, 0.11'

    if not os.path.exists('vis'): os.mkdir('vis')

  def visualizeTimestep(self, step, context, query, attn_t, x_t, a_t, M_t, b_t, out):
    if not os.path.exists('vis_tmp'): os.mkdir('vis_tmp')

    # Add sample/query node
    path = self._saveSample(step, context, query)
    self.G.add_node('sample_{}'.format(step), image=path)

    # Add input/output attention maps nodes
    path = self._saveAttention(step, 'input',  a_t)
    self.G.add_node('a_{}'.format(step), image=path)
    path = self._saveAttention(step, 'output', b_t)
    self.G.add_node('b_{}'.format(step), image=path)

    # Connect input to output with edges
    self.G.add_edge('a_{}'.format(step), 'b_{}'.format(step))
    self.G.add_edge('sample_{}'.format(step), 'b_{}'.format(step))

    # Add attn and text input nodes and edges
    self.G.add_node('x_{}'.format(step), labelloc='m',
                    label='attn_{}: {}\nx_{}: {}'.format(step, attn_t.cpu().numpy().squeeze(), step, x_t.cpu().numpy().squeeze()))
    self.G.add_edge('x_{}'.format(step), 'b_{}'.format(step))

    # Add output node and edge
    self.G.add_node('out_{}'.format(step), labelloc='m', label='out_{}: {}'.format(step, out.cpu().numpy().squeeze()))
    self.G.add_edge('a_{}'.format(step), 'out_{}'.format(step))

    # Add composition matrix node
    if step < self.comp_length - 1:
      self.G.add_node('M_{}'.format(step), labelloc='m', label='M_{}:\n{}'.format(step, M_t.cpu().numpy().transpose().squeeze()))

    # If not the first timestep connect input attention to past
    # compositional and output attention nodes
    if step > 0:
      self.G.add_edge('b_{}'.format(step-1), 'a_{}'.format(step))
      self.G.add_edge('M_{}'.format(step-1), 'a_{}'.format(step))

  def saveGraph(self, prefix=''):
    self.G.draw('vis/{}_forward_viz.png'.format(prefix), prog='dot')
    shutil.rmtree('vis_tmp')

  def _saveSample(self, step, sample, query):
    path = 'vis_tmp/sample_{}.png'.format(step)

    plt.figure()
    plt.title(self.query_lang.decodeQuery(query.cpu().squeeze()))
    plt.imshow(sample[0].cpu().permute(1,2,0), cmap='gray')
    plt.savefig(path)
    plt.close()

    return path

  def _saveAttention(self, step, attn_type, a):
    num_attention = a.size(1)
    path = 'vis_tmp/{}_attention_{}.png'.format(attn_type, step)

    # Plot each attention map with the module name as given
    fig, ax = plt.subplots(nrows=1, ncols=num_attention, figsize=(8, 4))
    for i, axi in enumerate(ax.flat):
      if attn_type == 'input':
        axi.set_title(self.module_input_names[i])
      else:
        axi.set_title(self.module_output_names[i])
      axi.imshow(a[0,i].cpu(), cmap='gray', vmin=0.0, vmax=1.0)

    # Save the figure to a temp file
    plt.tight_layout(True)
    plt.savefig(path)
    plt.close()

    return path
