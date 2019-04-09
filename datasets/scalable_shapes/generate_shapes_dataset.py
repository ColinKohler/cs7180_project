import argparse
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import skimage
import random
import tqdm
import ipdb as ipdb
import os

import query_ast as q_ast

def main(args):
  # Init empty dataset
  samples = np.empty((args.num_samples, args.sample_size, args.sample_size, 3))
  labels = np.empty((args.num_samples, 1))
  queries = list()

  # Grid specific
  num_cells = args.grid_size**2
  pixel_cell_size = int(args.sample_size / args.grid_size)

  # Generation junk
  shape_generators = {'ellipse': generateRandomEllipse, 'plus': generateRandomPlus,
                      'rectangle': generateRandomRect}
  num_true = 0

  for i in tqdm.trange(args.num_samples):
    not_retained = True
    while not_retained:
      # Generate sample grid
      sample = np.zeros((args.sample_size, args.sample_size, 3))

      # Add shapes to sample
      shape_masks = npr.rand(num_cells) > 0.5
      cell_length = list(range(args.grid_size))
      cell_indexes = np.dstack(np.meshgrid(cell_length, cell_length)).reshape(-1, 2)
      ast_grid = np.zeros((2, args.grid_size, args.grid_size))

      for cell_idx, mask in zip(cell_indexes, shape_masks):
        # Check to see if we will generate a shape in this cell
        if not mask:
          continue

        # Generate random shape of a random color
        color_str, color_rgb = generateRandomColor()
        shape_str, shape_generator = random.choice(list(shape_generators.items()))
        if args.aa:
          shape = generateAntiAliasedShape(shape_generator, pixel_cell_size, color_rgb)
        else:
          shape = shape_generator(pixel_cell_size, color_rgb)

        # Append shape into cell and cell sting into ast grid
        sample[cell_idx[1]*pixel_cell_size:(cell_idx[1]+1)*pixel_cell_size,
               cell_idx[0]*pixel_cell_size:(cell_idx[0]+1)*pixel_cell_size:, :] = shape
        ast_grid[:, cell_idx[1], cell_idx[0]] = [q_ast.COLOR_PROPERTY_INTS[color_str],
                                                 q_ast.SHAPE_PROPERTY_INTS[shape_str]]

      # Generate and compare query
      if args.query_gen_function == 1:
        query = generateFullRandomQuery(args.query_depth)
      elif args.query_gen_function == 2:
        query = generateRelationalAndQuery(args.query_depth)
      elif args.query_gen_function == 3:
        query = generateExistAndQuery(args.query_depth)
      else:
        query = generateQuery()
      label = getLabel(ast_grid, query)

      # Add sample to dataset
      too_many_trues = ((float(num_true - 5) / float(i+1) ) > 0.5)
      too_few_trues = ((float(num_true + 5) / float(i+1) ) < 0.5)
      do_not_retain = (too_many_trues and label) or (too_few_trues and (not label))
      if args.debug:
        print(i+1, "Samples Created.  Number True:", num_true, "Current Label:", label, "Retain?", not do_not_retain)
      if (not do_not_retain):
        samples[i] = sample
        queries.append(query.query(parens = args.parenthesis))
        labels[i] = label
        not_retained = False
        num_true += int(label)

      if args.debug:
        plt.title('{}: {}'.format(query.query(parens = args.parenthesis), label))
        plt.imshow(sample)
        plt.axis('off')
        plt.show()

  # Save dataset
  if not args.debug:
    path = './data/'+args.dataset_name
    if (not os.path.isdir(path)):
      os.makedirs(path)

    if args.overwrite:
      mode = "wb"
    else:
      mode = "xb"

    with open(path+'/samples.npy', mode = mode) as samples_f:
      np.save(samples_f, samples)
    with open(path+'/queries.npy', mode = mode) as queries_f:
      np.save(queries_f, queries)
    with open(path+'/labels.npy', mode = mode) as labels_f:
      np.save(labels_f, labels)
    print(i+1, "Samples Created.  Number True:", num_true)

#=============================================================================#
#                            Query Generation Functions                       #
#=============================================================================#

def generateQuery():
  ''' Generate query by combining all possible query parts '''
  # if npr.rand() > 0.1:
  # prop_1 = q_ast.generateRandomProperty()
  # prop_2 = q_ast.generateRandomProperty()
  # query = q_ast.Is(q_ast.generateRandomRelational(prop_1, prop_2))
  # else:
  query = q_ast.Is(q_ast.generateRandomProperty())

  return query

def generateFullRandomQuery(max_depth,
      ops = [q_ast.Property(),q_ast.Shape(),q_ast.Color(),q_ast.And(),q_ast.Or(),q_ast.Left()
      ,q_ast.Right(),q_ast.Above(),q_ast.Below()],
      prob = [0.2,0.15,0.15,0.13,0.13,0.06,0.06,0.06,0.06]):
  query = q_ast.Is()
  query.sample(max_depth-1, ops, prob)
  return query

def generateRelationalAndQuery(len):
  prop_1 = q_ast.generateRandomNullary(0.15)
  prop_2 = q_ast.generateRandomNullary(0.15)
  if (npr.rand() < 0.33):
    randRelational = q_ast.Is(q_ast.generateRandomRelational(prop_1, prop_2))
  else:
    randRelational = q_ast.Is(prop_1)
  if len < 2:
    return randRelational
  else:
    return q_ast.BoolAnd(randRelational, generateRelationalAndQuery(len-1))

def generateExistAndQuery(len):
  randProp = q_ast.Is(q_ast.generateRandomNullary())
  if len < 2:
    return randRelational
  else:
    return q_ast.BoolAnd(randProp, generateRelationalAndQuery(len-1))

def getLabel(str_sample, query):
  ''' Parse query and use str_sample to determine label '''
  return query.eval(str_sample)

#=============================================================================#
#                            Sample Generation Functions                      #
#=============================================================================#

def generateAntiAliasedShape(shape_generator, cell_size, color):
  ''' Generate anti aliased ellipse in the desired grid_cell '''
  large_img = shape_generator(4*cell_size, color)
  return skimage.transform.resize(large_img, (cell_size,cell_size),
                                  mode='constant', anti_aliasing=True)

def generateRandomPlus(cell_size, color):
  ''' Generate plus in the desired grid_cell '''
  default_width = cell_size / 8
  default_height = cell_size / 2
  center = cell_size / 2

  # Generate center, width, and height with noise
  cx = int(center + int(npr.normal(0, cell_size/16)))
  cy = int(center + int(npr.normal(0, cell_size/16)))
  width = int(default_width + int(npr.normal(0, cell_size/24)))
  height = int(default_height + int(npr.normal(0, cell_size/12)))

  # Draw the plus and set its color
  cell_img = np.zeros((cell_size, cell_size, 3))
  rr1, cc1 = skimage.draw.rectangle([int(cx-height/2), int(cy-width/2)],
                                    extent=[height, width],
                                    shape=cell_img.shape[:2])
  rr2, cc2 = skimage.draw.rectangle([int(cx-width/2), int(cy-height/2)],
                                    extent=[width, height],
                                    shape=cell_img.shape[:2])
  cell_img[rr1,cc1,:] = color
  cell_img[rr2,cc2,:] = color

  return cell_img

def generateRandomEllipse(cell_size, color):
  ''' Generate ellipse in the desired grid_cell '''
  default_radius = cell_size / 3
  center = cell_size / 2

  # Generate center and radii with noise. Clipping to ensure full shape is in cell
  cx = np.clip(center + int(npr.normal(0, cell_size/16)),
               cell_size * ((1/2) - (1/16)),
               cell_size * ((1/2) + (1/16)))
  cy = np.clip(center + int(npr.normal(0, cell_size/16)),
               cell_size * ((1/2) - (1/16)),
               cell_size * ((1/2) + (1/16)))
  r_major = np.clip(default_radius + int(npr.normal(0, cell_size/12)),
                    cell_size * ((1/3) - (1/12)),
                    cell_size * ((1/3) + (1/12)))
  r_minor = np.clip(default_radius + int(npr.normal(0, cell_size/12)),
                    cell_size * ((1/3) - (1/12)),
                    cell_size * ((1/3) + (1/12)))


  # Draw the ellipse and set its color
  cell_img = np.zeros((cell_size, cell_size, 3))
  rr, cc = skimage.draw.ellipse(cx, cy, r_major, r_minor, shape=cell_img.shape[:2])
  cell_img[rr,cc,:] = color

  return cell_img

def generateRandomRect(cell_size, color):
  default_width = cell_size / 2
  default_height = cell_size / 2
  center = cell_size / 2

  # Generate center, width, and height with noise
  cx = int(center + int(npr.normal(0, cell_size/12)))
  cy = int(center + int(npr.normal(0, cell_size/12)))
  width = int(default_width + int(npr.normal(0, cell_size/8)))
  height = int(default_height + int(npr.normal(0, cell_size/8)))

  # Draw the plus and set its color
  cell_img = np.zeros((cell_size, cell_size, 3))
  rr, cc = skimage.draw.rectangle([int(cx-height/2), int(cy-width/2)],
                                    extent=[height, width],
                                    shape=cell_img.shape[:2])
  cell_img[rr,cc,:] = color

  return cell_img

def generateRandomColor(noise_mean=0.15, noise_var=3):
  ''' Generates a random color '''
  colors = getColorsDict()
  color_key, color = random.choice(list(colors.items()))
  color = np.clip(color + npr.normal(0, noise_mean, noise_var), 0, 1)
  return color_key, color

def getColorsStr():
  ''' Gets the strings of all the colors we currently use '''
  return getColorsDict().keys()

def getColorsDict():
  ''' Gets the dictonary of all the colors we currently use '''
  return {'red': np.array([1., 0., 0.]), 'green': np.array([0., 1., 0.]),
          'blue': np.array([0., 0., 1.0])}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_samples', type=int,
      help='Number of samples to generate')
  parser.add_argument('--grid_size', type=int, default=2,
      help='Size of the grid (default=2x2)')
  parser.add_argument('--sample_size', type=int, default=64,
      help='Sample size in pixels (64x64)')
  parser.add_argument('--aa', default=False, action='store_true',
      help='Use anti aliasing when generating shapes')
  parser.add_argument('--debug', default=False, action='store_true',
      help='Enter debugging mode')
  parser.add_argument('--dataset_name', default='v4', type=str,
      help='Name of dataset folder.  Will be stored at data/DATASET_NAME/*')
  parser.add_argument('--query_gen_function', type=int, default=0,
      help='Query generator function: 0=generate exist queries, 1=generate full random queries, 2=generate relational-And queries, 3=generate exist And queries')
  parser.add_argument('--query_depth', type=int, default=2,
      help='Maximum recursion depth in query AST')
  parser.add_argument('--parenthesis', default=False, action='store_true',
      help='Output parenthesis in queries')
  parser.add_argument('--overwrite', default=False, action='store_true',
      help='Overwrites dataset if it already exists')
  parser.add_argument('--balance', default=False, action='store_true',
      help='try to keep True/False ratio balanced.')

  args = parser.parse_args()
  main(args)
