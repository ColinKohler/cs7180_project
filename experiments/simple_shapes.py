import os
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import tqdm
import ipdb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('../')
import dataset_loader
from rnmn import RNMN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config):
  # Load dataset
  query_lang, train_loader, test_loader = dataset_loader.createScalableShapesDataLoader(config.dataset, batch_size=config.batch_size)

  # Init model
  model = RNMN(query_lang.num_words, config.hidden_size, device, config.mt_norm, config.comp_length, config.comp_stop).to(device)
  if config.weight_decay == 0:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
  criterion = nn.NLLLoss()

  # Create TQDM progress bar
  pbar = tqdm.tqdm(total=config.epochs)
  pbar.set_description('Train Loss:0.0 | Train Acc:0.0 | Test Loss:0.0 | Test Acc:0.0')

  train_losses, test_losses, test_accs = list(), list(), list()
  for epoch in range(config.epochs):
    # Train for a single epoch iterating over the minibatches
    train_loss = 0
    #config.debug = False
    #if(epoch > 3):
    #  config.debug = True
    for samples, queries, query_lens, labels in train_loader:
      train_loss += trainBatch(model, optimizer, criterion, samples, queries,
                               query_lens, labels, debug=config.debug)

    # Test for a single epoch iterating over the minibatches
    test_loss, test_correct = 0, 0
    for i, (samples, queries, query_lens, labels) in enumerate(test_loader):
      _, batch_loss, batch_correct = testBatch(model, criterion, samples, queries,
                                               query_lens, labels, debug=config.debug)
      test_loss += batch_loss
      test_correct += batch_correct

    # Bookkeeping
    train_losses.append(train_loss / (len(train_loader.dataset) / config.batch_size))
    test_losses.append(test_loss / (len(test_loader.dataset) / config.batch_size))
    test_accs.append(test_correct / len(test_loader.dataset))

    # Update progress bar
    pbar.set_description('Train Loss:{:.5f} | Test Loss:{:.5f} | Test Acc:{:.3f}'.format(
      train_losses[-1], test_losses[-1], test_accs[-1]))
    pbar.update(1)

  # Close progress bar
  pbar.close()

  samples, queries, query_lens, labels = test_loader.dataset[:1028]
  # for query in queries:
  #   print(' '.join(query_lang.decodeQuery(query)))
  # plt.title(query_lang.decodeQuery(queries[0]))
  # plt.imshow(samples[0].permute(1,2,0))
  # plt.show()
  output, loss, correct = testBatch(model, criterion, samples, queries, query_lens, labels, debug=False)
  # print(output.argmax(dim=1).cpu())
  # print(labels.round().t().long().cpu().squeeze())
  print(correct)

def test(config):
  pass

def trainBatch(model, optimizer, criterion, samples, queries, query_lens, labels, debug=False):
  model.train()
  # Transfer data to gpu/cpu and pass through model
  samples, queries, query_lens, labels = tensorToDevice(samples, queries, query_lens, labels)
  output = model(queries, query_lens, samples, debug=debug)

  # Compute loss & step optimzer
  optimizer.zero_grad()
  loss = criterion(output, labels.squeeze().long())
  loss.backward()
  optimizer.step()

  return loss.item()

def testBatch(model, criterion, samples, queries, query_lens, labels, debug=False):
  model.eval()
  with torch.no_grad():
    # Transfer data to gpu/cpu and pass through model
    samples, queries, query_lens, labels = tensorToDevice(samples, queries, query_lens, labels)
    output = model(queries, query_lens, samples, debug=debug)

    # Compute loss & accuracy
    loss = criterion(output, labels.squeeze(1).long())
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(labels.view_as(pred).round().long()).sum()

  return output, loss.item(), correct.item()

def tensorToDevice(*tensors):
  return [tensor.to(device) for tensor in tensors]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('epochs', type=int, default=100,
      help='Number of epochs to train/test for')
  parser.add_argument('dataset', type=str, default='v1',
      help='folder to read in dataset from')
  parser.add_argument('--lr', type=float, default=1e-3,
      help='Learning rate for the model')
  parser.add_argument('--weight_decay', type=float, default=1e-4,
      help='L2 weight decay parameter for optimizr')
  parser.add_argument('--hidden_size', type=int, default=256,
      help='Number of units in the LSTM layers in the query encoder/decoders')
  parser.add_argument('--batch_size', type=int, default=256,
      help='Minibatch size for data loaders')
  parser.add_argument('--seed', type=int, default=None,
      help='PyTorch random seed to use for this run')
  parser.add_argument('--load_model', type=str, default=None,
      help='Model to load')
  parser.add_argument('--test', default=False, action='store_true',
      help='Enter test mode. Must specify a model to load')
  parser.add_argument('--debug', default=False, action='store_true',
      help='Enter debugging mode')
  parser.add_argument('--mt_norm', type=int, default=1,
      help='M_t matrix normalization (0=None, 1=row softmax (default), 2=2dsoftmax, 3=rowsum clamped [0,1]')
  parser.add_argument('--comp_length', type=int, default=5,
      help='maximum number of compositions (default = 5)')
  parser.add_argument('--comp_stop', type=int, default=1,
      help='method for selecting output timestep (0=last, 1=learned_weighted_avg (default))')

  args = parser.parse_args()
  if args.seed: torch.manual_seed(args.seed) # 9=good
  if not args.test:
    train(args)
  elif args.test and args.load_model:
    test(args)
  else:
    ValueError('Must specify model to load if in test mode.')
