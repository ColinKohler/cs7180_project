import argparse
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import skimage
import random

def main(args):
  # Init empty dataset
  samples = np.empty((args.num_samples, args.sample_size, args.sample_size, 3))
  labels = np.empty((args.num_samples, 1))
  queries = list()

  # Grid specific
  num_cells = args.grid_size**2
  pixel_cell_size = int(args.sample_size / args.grid_size)

  for i in range(args.num_samples):
    # Generate sample grid
    sample = np.zeros((args.sample_size, args.sample_size, 3))

    # Add shapes to sample
    shape_masks = npr.rand(num_cells) > 0.5
    cell_length = list(range(args.grid_size))
    cell_indexes = np.dstack(np.meshgrid(cell_length, cell_length)).reshape(-1, 2)
    string_grid = np.empty((args.grid_size, args.grid_size), dtype=np.object)

    for cell_idx, mask in zip(cell_indexes, shape_masks):
      # Check to see if we will generate a shape in this cell
      if not mask:
        continue

      # Generate random shape of a random color
      shape_generators = {'ellipse': generateRandomEllipse, 'plus': generateRandomPlus}

      color_str, color_rgb = generateRandomColor()
      shape_str, shape_generator = random.choice(list(shape_generators.items()))
      if args.aa:
        shape = generateAntiAliasedShape(shape_generator, pixel_cell_size, color_rgb)
      else:
        shape = shape_generator(pixel_cell_size, color_rgb)

      # Append shape into cell and cell sting into string grid
      sample[cell_idx[1]*pixel_cell_size:(cell_idx[1]+1)*pixel_cell_size,
             cell_idx[0]*pixel_cell_size:(cell_idx[0]+1)*pixel_cell_size:, :] = shape
      string_grid[cell_idx[1], cell_idx[0]] = '{}_{}'.format(color_str, shape_str)

    # Generate and compare query to sample to get label
    query = generateQuery(string_grid)
    label = getLabel(string_grid, query)

    # Add sample to dataset
    samples[i] = sample

    if args.debug:
      plt.imshow(sample)
      plt.show()

  # Save dataset
  if not args.debug:
    np.save('./datasets/simple_shapes/samples.npy', samples)
    np.save('./datasets/simple_shapes/queries.npy', queries)
    np.save('./datasets/simple_shapes/labels.npy', labels)

def generateQuery(grid):
  return None

def getLabel(sample, query):
  return None

def generateAntiAliasedShape(shape_generator, cell_size, color):
  ''' Generate anti aliased ellipse in the desired grid_cell '''
  large_img = shape_generator(4*cell_size, color)
  return skimage.transform.resize(large_img, (cell_size,cell_size), mode='constant', anti_aliasing=True)

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

def generateRandomColor(noise_mean=0.15, noise_var=3):
  ''' Generates a random color'''
  colors = {'red': np.array([1., 0., 0.]), 'green': np.array([0., 1., 0.]),
            'blue': np.array([0., 0., 1.0])}
  color_key, color = random.choice(list(colors.items()))
  color = np.clip(color + npr.normal(0, noise_mean, noise_var), 0, 1)
  return color_key, color

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

  args = parser.parse_args()
  main(args)
