import os
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data_utils

DATASETS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/')
SCALABLE_SHAPES_PATH = os.path.join(DATASETS_PATH, 'scalable_shapes/data/')

class QueryLang(object):
  def __init__(self):
    self.word_to_index = dict()
    self.index_to_word = dict()
    self.word_to_count = dict()
    self.num_words = 0

  def addQuery(self, query):
    for word in query.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word_to_index:
      self.word_to_index[word] = self.num_words
      self.word_to_count[word] = 1
      self.index_to_word[self.num_words] = word
      self.num_words += 1
    else:
      self.word_to_count[word] += 1

  def encodeQuery(self, query):
    return [self.word_to_index[word] for word in query.split(' ')]

def createScalableShapesDataLoader(dataset, batch_size=64, rebalanced=True):
  # Load labels, queries, and samples as numpy arrays
  if rebalanced:
    labels = np.load(os.path.join(SCALABLE_SHAPES_PATH, dataset, 'rebalanced_labels.npy'))
    queries = np.load(os.path.join(SCALABLE_SHAPES_PATH, dataset, 'rebalanced_queries.npy'))
    samples = np.load(os.path.join(SCALABLE_SHAPES_PATH, dataset, 'rebalanced_samples.npy'))
  else:
    labels = np.load(os.path.join(SCALABLE_SHAPES_PATH, dataset, 'labels.npy'))
    queries = np.load(os.path.join(SCALABLE_SHAPES_PATH, dataset, 'queries.npy'))
    samples = np.load(os.path.join(SCALABLE_SHAPES_PATH, dataset, 'samples.npy'))

  # Create the query language
  query_lang = QueryLang()
  encoded_queries = list()
  for query in queries:
    query_lang.addQuery(query)
    encoded_queries.append(query_lang.encodeQuery(query))
  encoded_queries = np.array(encoded_queries)

  # Split the data into a train/test datasets
  dataset_length = labels.shape[0]
  indices = npr.permutation(dataset_length)
  train_idx, test_idx = indices[:int(dataset_length*.8)], indices[int(dataset_length*.8):]

  train_labels, train_queries, train_samples = labels[train_idx], encoded_queries[train_idx], samples[train_idx]
  test_labels, test_queries, test_samples = labels[test_idx], encoded_queries[test_idx], samples[test_idx]

  # Load data into Torch tensors
  train_labels, train_queries, train_samples = convertNumpyToTorch(train_labels, train_queries, train_samples)
  test_labels, test_queries, test_samples = convertNumpyToTorch(test_labels, test_queries, test_samples)

  # Load data into dataset loaders
  train_dataset = data_utils.TensorDataset(train_samples, train_queries, train_labels)
  train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = data_utils.TensorDataset(test_samples, test_queries, test_labels)
  test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  return query_lang, train_loader, test_loader

def convertNumpyToTorch(*arrays):
  return [torch.from_numpy(array) for array in arrays]
