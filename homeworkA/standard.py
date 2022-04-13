import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.scatter(z[:, 0], z[:, 1], s=50, c=color, cmap="Set2")
    plt.show()


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# number of nodes
nTrainingNodes = np.count_nonzero(dataset.data.train_mask)
nTestNodes = np.count_nonzero(dataset.data.test_mask)
nValNodes = np.count_nonzero(dataset.data.val_mask)
# semi-supervised learning can be used for node-level classification, which is basically
# identifying the unlabeled nodes in the graph.
# supervised learning can be used for graph-level classification where we try to predict
# the node labels for the entire graph.


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


model = MLP(hidden_channels=16)
data = dataset[0]
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


# training with early stop
previous_loss = 1e6
n = 0
for epoch in range(1, 201):
    new_loss = train()
    if previous_loss <= new_loss:
        print(f'Epoch: {epoch:03d}, Loss: {previous_loss:.4f}')

        # count
        n += 1
        if n == 2:
            break
    else:
        previous_loss = new_loss

        # reset
        n = 0

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
visualize(model(data.x), color=data.y)




