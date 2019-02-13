import argparse
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import skimage

NUM_COLORS = 2


def main(args):
  # Dataset
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
      # if mask generate shape randomly
      # if npr.rand() > 0.5:
      #   shape = generatePlus()
      # else:
      shape = generateAntiAliasedRandomEllipse(pixel_cell_size)

      # Append shape into cell
      sample[cell_idx[1]*pixel_cell_size:(cell_idx[1]+1)*pixel_cell_size,
             cell_idx[0]*pixel_cell_size:(cell_idx[0]+1)*pixel_cell_size:, :] = shape


    # Generate and compare query to sample to get label
    plt.imshow(sample, cmap='gray')
    plt.show()


def generateQuery():
  pass

def getLabel(sample, query):
  pass

def generatePlus(grid_cell, transform, color):
  ''' Generate plus in the desired grid_cell '''
  pass

def generateRandomColor():
  ''' Generate one of NUM_COLORS colors plus some random noise'''
  # Colors from HTML 4 sorted by commonness of names
  colors = {"red":(1,0,0),"blue":(0,0,1),"green":(0,.5,0),"purple":(.5,0,.5),"yellow":(1,1,0),"aqua":(0,1,1),"orange":(1,0.65,0),"gray":(.5,.5,.5),"teal":(0,.5,.5),"navy":(0,0,.5),"lime":(0,1,0),"fuchsia":(1,0,1),"olive":(.5,.5,0),"maroon":(.5,0,0)}
  # Truncate to NUM_COLORS
  colors = dict(list(colors.items())[0:NUM_COLORS])
  color_key_list = list(colors)
  color_indx = npr.randint(0,NUM_COLORS)
  center_RGB_color = list(colors[color_key_list[color_indx]])
  noise = npr.normal(0, .15, 3)
  return np.clip(center_RGB_color + noise,0,1)
  
def generateAntiAliasedRandomEllipse(cell_size):
  ''' Generate anti aliased ellipse in the desired grid_cell '''
  large_img = generateRandomEllipse(4*cell_size)
  cell_img = skimage.transform.resize(large_img,(cell_size,cell_size))
  return cell_img
  
def generateRandomEllipse(cell_size):
  ''' Generate ellipse in the desired grid_cell '''
  default_radius = cell_size / 3
  center = cell_size / 2
  color = generateRandomColor()

  cx = np.clip(center + int(npr.normal(0, cell_size/16)), cell_size * ((1/2) - (1/16)), cell_size * ((1/2) + (1/16)))
  cy = np.clip(center + int(npr.normal(0, cell_size/16)), cell_size * ((1/2) - (1/16)), cell_size * ((1/2) + (1/16)))
  r_major = np.clip(default_radius + int(npr.normal(0, cell_size/12)), cell_size * ((1/3) - (1/12)), cell_size * ((1/3) + (1/12)))
  r_minor = np.clip(default_radius + int(npr.normal(0, cell_size/12)), cell_size * ((1/3) - (1/12)), cell_size * ((1/3) + (1/12)))

  cell_img = np.zeros((cell_size, cell_size, 3))
  rr, cc = skimage.draw.ellipse(cx, cy, r_major, r_minor)
  cell_img[rr,cc,:] = color
  
  return cell_img

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_samples', type=int,
      help='Number of samples to generate')
  parser.add_argument('--grid_size', type=int, default=2,
      help='Size of the grid (default=2x2)')
  parser.add_argument('--sample_size', type=int, default=64,
      help='Sample size in pixels (64x64)')

  args = parser.parse_args()
  main(args)
