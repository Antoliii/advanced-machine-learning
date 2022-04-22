import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time


def visualize(axs_, h, color, title_):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    axs_.scatter(z[:, 0], z[:, 1], s=4, c=color, cmap='Set2')
    axs_.set_title(title_)


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset.data

# Number of nodes
nTrainingNodes = np.count_nonzero(data.train_mask)
nTestNodes = np.count_nonzero(data.test_mask)
nValNodes = np.count_nonzero(data.val_mask)
print(f'Training nodes: {nTrainingNodes}\nTest nodes: {nTestNodes}\nValidation nodes: {nValNodes}\n')
# semi-supervised learning can be used for node-level classification, which is basically
# identifying the unlabeled nodes in the graph.
# supervised learning can be used for graph-level classification where we try to predict
# the node labels for the entire graph.


# supervised learning standard dense network
class NN(torch.nn.Module):
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


# semi-supervised learning convolutional network
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# semi-supervised graph attention network
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads, dropout):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# functions for training and testing
def train(model_, semi_supervised, optimizer_):
    model_.train()
    optimizer_.zero_grad()

    if semi_supervised:
        out = model_(data.x, data.edge_index)
    else:
        out = model_(data.x)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # derive gradients.
    optimizer_.step()  # update parameters based on gradients.
    return loss, model_, optimizer_


def test(model_, semi_supervised):
    model_.eval()

    if semi_supervised:
        out = model_(data.x, data.edge_index)
    else:
        out = model_(data.x)

    predicts = out.argmax(dim=1)
    test_correct = predicts[data.test_mask] == data.y[data.test_mask]
    test_accuracy = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_accuracy


modelNN = NN(hidden_channels=16)
modelGCN = GCN(hidden_channels=16)
modelGAT = GAT(hidden_channels=16, heads=3, dropout=0.7)
models = [modelNN, modelGCN, modelGAT]
semiSupervised = dict(zip([modelNN, modelGCN, modelGAT], [False, True, True]))


# early stop
r = 0
fig, axs = plt.subplots(3, 2)
for model in models:
    previousLoss = 1e6
    n = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    losses = []
    for epoch in range(1, 999):
        newLoss, model, optimizer = train(model_=model, semi_supervised=semiSupervised[model], optimizer_=optimizer)
        losses.append(newLoss.item())
        if previousLoss <= newLoss.item():
            print(f'Model: {model.__class__.__name__}, Epoch: {epoch:03d}, Loss: {previousLoss:.4f}')

            # count
            n += 1
            if n == 10:
                testAcc = test(model, semi_supervised=semiSupervised[model])
                print(f'Model: {model.__class__.__name__}, Test accuracy: {100*testAcc:.2f}%\n')
                time.sleep(1)
                break
        else:
            n = 0
            previousLoss = newLoss.item()

    # visualize
    if not semiSupervised[model]:

        visualize(axs_=axs[r, 0], h=model(data.x), color=data.y, title_=model.__class__.__name__)
        axs[r, 1].plot(losses)
        axs[r, 1].set_title('Loss')
    else:
        visualize(axs_=axs[r, 0], h=model(data.x, data.edge_index), color=data.y, title_=model.__class__.__name__)
        axs[r, 1].plot(losses)
        axs[r, 1].set_title('Loss')
    r += 1
plt.subplots_adjust(hspace=0.4)
plt.show()
