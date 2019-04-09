import numpy as np
import numpy.random as npr
import argparse
import os

def main(args):
  samples1 = np.load(args.dataset1+'samples.npy')
  queries1 = np.load(args.dataset1+'queries.npy')
  labels1 = np.load(args.dataset1+'labels.npy')
  samples2 = np.load(args.dataset2+'samples.npy')
  queries2 = np.load(args.dataset2+'queries.npy')
  labels2 = np.load(args.dataset2+'labels.npy')

  path = args.output_dataset
  if path[-1] == '/':
    path = path[:-1]
  if (not os.path.isdir(path)):
    os.makedirs(path)

  if args.overwrite:
    mode = "wb"
  else:
    mode = "xb"

  with open(path+'/samples.npy', mode = mode) as samples_f:
    np.save(samples_f, np.concatenate((samples1,samples2)))
  with open(path+'/queries.npy', mode = mode) as queries_f:
    np.save(queries_f, np.concatenate((queries1,queries2)))
  with open(path+'/labels.npy', mode = mode) as labels_f:
    np.save(labels_f, np.concatenate((labels1,labels2)))




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset1', type=str,
      help='First Dataset Include full prefix up to samples,queries.  E.g. /data/v2/ (with slash) or /data/v2/rebalanced_')
  parser.add_argument('dataset2', type=str,
      help='Second Dataset')
  parser.add_argument('output_dataset', type=str,
      help='Path of new combined dataset')
  parser.add_argument('--overwrite', default=False, action='store_true',
      help='Overwrites dataset if it already exists')
  args = parser.parse_args()
  main(args)