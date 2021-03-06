import os
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data_utils

DATASETS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
SCALABLE_SHAPES_PATH = os.path.join(DATASETS_PATH, 'scalable_shapes','data')

class QueryLang(object):
  def __init__(self):
    self.EOQ_token = 0
    self.word_to_index = dict()
    self.index_to_word = {self.EOQ_token: 'EOQ'}
    self.word_to_count = dict()
    self.num_words = 1
    self.max_len = 0

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

  def decodeQuery(self, encoded_query):
    return [self.index_to_word[index.item()] for index in encoded_query]

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

  # Image tensors for pyTorch are Batch X Channels X X_dim X Y_dim.
  # NP images are currently Batch X X_dim X Y_dim X Channels
  samples = np.transpose(samples,(0,3,1,2))
  samples -= samples.mean(axis=0)
  samples = samples.astype(np.float32) 
  
  (samples - samples.mean(axis=0)).astype(np.float32)

  # Create the query language
  query_lang = QueryLang()
  encoded_queries = list()
  query_lens = list()
  max_len = 0
  for query in queries:
    query_lang.addQuery(query)
    encoded_query = query_lang.encodeQuery(query)

    max_len = max(max_len, len(encoded_query))
    encoded_queries.append(encoded_query)
    query_lens.append(len(encoded_query))

  query_lang.max_len = max_len

  # Pad encoded queries with EOQ tokens
  queries = np.ones((len(query_lens), max_len), dtype=np.long) * query_lang.EOQ_token
  query_lens = np.array(query_lens, dtype=np.long)
  for i, q in enumerate(encoded_queries):
    queries[i,:len(q)] = q

  # Split the data into a train/test datasets
  dataset_length = labels.shape[0]
  indices = npr.permutation(dataset_length)
  train_idx, test_idx = indices[:int(dataset_length*.8)], indices[int(dataset_length*.8):]

  train_labels, train_samples = labels[train_idx],  samples[train_idx]
  train_queries, train_query_lens = queries[train_idx], query_lens[train_idx]

  test_labels, test_samples = labels[test_idx], samples[test_idx]
  test_queries, test_query_lens = queries[test_idx], query_lens[test_idx]

  # Load data into Torch tensors
  train_labels, train_samples = convertNumpyToTorch(train_labels, train_samples)
  train_queries, train_query_lens = convertNumpyToTorch(train_queries, train_query_lens)
  train_queries, train_query_lens = train_queries.long(), train_query_lens.long()
  train_samples = train_samples.float()

  test_labels, test_samples = convertNumpyToTorch(test_labels, test_samples)
  test_queries, test_query_lens = convertNumpyToTorch(test_queries, test_query_lens)
  test_queries, test_query_lens = test_queries.long(), test_query_lens.long()
  test_samples = test_samples.float()

  # Load data into dataset loaders
  train_dataset = data_utils.TensorDataset(train_samples, train_queries, train_query_lens, train_labels)
  train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = data_utils.TensorDataset(test_samples, test_queries, test_query_lens, test_labels)
  test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  return query_lang, train_loader, test_loader, max_len

def convertNumpyToTorch(*arrays):
  return [torch.from_numpy(array) for array in arrays]
