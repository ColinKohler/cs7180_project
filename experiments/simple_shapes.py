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
torch.manual_seed(0) # 9=good

def tensorToDevice(*tensors):
  return [tensor.to(device) for tensor in tensors]

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

    # Compute loss & acccriterionuracy
    loss = criterion(output, labels.squeeze(1).long())
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(labels.view_as(pred).round().long()).sum()

  return output, loss.item(), correct.item()

def train():
  # Set hyperparams and load dataset
  lr = 1e-4
  hidden_size = 256
  #overliberal use of squeeze prevents setting to 1
  batch_size = 256
  epochs = 10

  query_lang, train_loader, test_loader = dataset_loader.createScalableShapesDataLoader('v3', batch_size=batch_size)

  # Init model
  model = RNMN(query_lang.num_words, hidden_size, device).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.NLLLoss()

  # Create TQDM progress bar
  pbar = tqdm.tqdm(total=epochs)
  pbar.set_description('Train Loss:0.0 | Train Acc:0.0 | Test Loss:0.0 | Test Acc:0.0')

  train_losses, test_losses, test_accs = list(), list(), list()
  for epoch in range(epochs):
	  # Train for a single epoch iterating over the minibatches
	  train_loss = 0
	  for samples, queries, query_lens, labels in train_loader:
	    train_loss += trainBatch(model, optimizer, criterion, samples, queries, query_lens, labels)

	  # Test for a single epoch iterating over the minibatches
	  test_loss, test_correct = 0, 0
	  for samples, queries, query_lens, labels in test_loader:
	    _, batch_loss, batch_correct = testBatch(model, criterion, samples, queries, query_lens, labels)
	    test_loss += batch_loss
	    test_correct += batch_correct

	  # Bookkeeping
	  train_losses.append(train_loss / (len(train_loader.dataset) / batch_size))
	  test_losses.append(test_loss / (len(test_loader.dataset) / batch_size))
	  test_accs.append(test_correct / len(test_loader.dataset))

	  # Update progress bar
	  pbar.set_description('Train Loss:{:.5f} | Test Loss:{:.5f} | Test Acc:{:.3f}'.format(
      train_losses[-1], test_losses[-1], test_accs[-1]))
	  pbar.update(1)

  # Close progress bar
  pbar.close()

  samples, queries, query_lens, labels = test_loader.dataset[:1]
  plt.title(query_lang.decodeQuery(queries[0]))
  plt.imshow(samples[0].permute(1,2,0))
  plt.show()
  output, loss, correct = testBatch(model, criterion, samples, queries, query_lens, labels, debug=False)
  print(output)
  print(correct)

if __name__ == '__main__':
  # parser = argparse.Ark

  train()
