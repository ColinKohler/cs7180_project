import numpy as np
import numpy.random as npr

def main():
  samples = np.load('data/v4/samples.npy')
  queries = np.load('data/v4/queries.npy')
  labels = np.load('data/v4/labels.npy')

  # true_relation = list();
  # true_exists = list()
  # for true_index in np.where(labels)[0]:
  #   if 'above' in queries[true_index] or 'below' in queries[true_index] or 'left' in queries[true_index] or 'right' in queries[true_index]:
  #     true_relation.append(true_index)
  #   else:
  #     true_exists.append(true_index)

  # min_subset = min(len(true_relation), len(true_exists), np.sum(np.logical_not(labels)))
  # relation_subset = npr.choice(true_relation, min_subset)
  # exists_subset = npr.choice(true_exists, min_subset)
  # false_subset = npr.choice(np.where(np.logical_not(labels))[0], min_subset*2)
  # min_subset = min(len(true_relation), len(true_exists), np.sum(np.logical_not(labels)))
  # relation_subset = npr.choice(true_relation, min_subset)
  # exists_subset = npr.choice(true_exists, min_subset)
  # false_subset = npr.choice(np.where(np.logical_not(labels))[0], min_subset*2)

  # np.save('data/v4/rebalanced_samples.npy', samples[np.concatenate((relation_subset, exists_subset, false_subset))])
  # np.save('data/v4/rebalanced_queries.npy', queries[np.concatenate((relation_subset, exists_subset, false_subset))])
  # np.save('data/v4/rebalanced_labels.npy', labels[np.concatenate((relation_subset, exists_subset, false_subset))])

  exists_subset = np.where(labels)[0]
  false_subset = npr.choice(np.where(np.logical_not(labels))[0], len(exists_subset))

  np.save('data/v4/rebalanced_samples.npy', samples[np.concatenate((exists_subset, false_subset))])
  np.save('data/v4/rebalanced_queries.npy', queries[np.concatenate((exists_subset, false_subset))])
  np.save('data/v4/rebalanced_labels.npy', labels[np.concatenate((exists_subset, false_subset))])

if __name__ == '__main__':
  main()
