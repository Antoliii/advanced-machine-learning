import torch
import numpy as np
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/Planetoid', name='Cora')

# number of nodes
nTrainingNodes = np.count_nonzero(dataset.data.train_mask)
nTestNodes = np.count_nonzero(dataset.data.test_mask)
nValNodes = np.count_nonzero(dataset.data.val_mask)
# semi-supervised learning can be used for node-level classification, which is basically
# identifying the unlabeled nodes in the graph.
# supervised learning can be used for graph-level classification where we try to predict
# the node labels for the entire graph.

