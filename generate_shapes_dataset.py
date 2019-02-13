import argparse
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import skimage

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
    cells = np.dstack(np.meshgrid(cell_length, cell_length)).reshape(-1, 2)

    for cell_idx, mask in zip(cells, shape_masks):
      # Check to see if we will generate a shape in this cell
      if not mask:
        continue

      # Generate random shape of a random color
      color_generators = [generateRandomRed, generateRandomGreen]
      shape_generators = [generateRandomEllipse]

      color_str, color_rgb = npr.choice(color_generators)()
      shape = npr.choice(shape_generators)(pixel_cell_size, color_rgb)

      # Append shape into cell
      sample[cell_idx[1]*pixel_cell_size:(cell_idx[1]+1)*pixel_cell_size,
             cell_idx[0]*pixel_cell_size:(cell_idx[0]+1)*pixel_cell_size:, :] = shape

    # Generate and compare query to sample to get label

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

def generateQuery():
  pass

def getLabel(sample, query):
  pass

def generateRandomPlus(cell_size, color_generators):
  ''' Generate plus in the desired grid_cell '''
  pass

def generateAntiAliasedRandomEllipse(cell_size):
  ''' Generate anti aliased ellipse in the desired grid_cell '''
  large_img = generateRandomEllipse(4*cell_size)
  cell_img = skimage.transform.resize(large_img,(cell_size,cell_size))
  return cell_img

def generateRandomEllipse(cell_size, color):
  ''' Generate ellipse in the desired grid_cell '''
  default_radius = cell_size / 3
  center = cell_size / 2

  # Clip the center and radii so they fit in the cell
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
  rr, cc = skimage.draw.ellipse(cx, cy, r_major, r_minor)
  cell_img[rr,cc,:] = color

  return cell_img

# TODO: This should generate a random shade of red
def generateRandomRed():
  ''' Generates a random shade of red '''
  color = np.clip(np.array([1., 0., 0.]) + npr.normal(0, .15, 3), 0, 1)
  return 'red', color

# TODO: This should generate a random shade of green
def generateRandomGreen():
  ''' Generates a random shade of green '''
  color = np.clip(np.array([0., 1., 0.]) + npr.normal(0, .15, 3), 0, 1)
  return 'green', color

# TODO: This should generate a random shade of blue
def generateRandomBlue():
  ''' Generates a random shade of blue '''
  color = np.clip(np.array([0., 0., 1.]) + npr.normal(0, .15, 3), 0, 1)
  return 'blue', color

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_samples', type=int,
      help='Number of samples to generate')
  parser.add_argument('--grid_size', type=int, default=2,
      help='Size of the grid (default=2x2)')
  parser.add_argument('--sample_size', type=int, default=64,
      help='Sample size in pixels (64x64)')
  parser.add_argument('--debug', default=False, action='store_true',
      help='Enter debugging mode')

  args = parser.parse_args()
  main(args)
