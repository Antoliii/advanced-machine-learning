import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

data = torch.load('data/graph_list_2022.pt')
print(data[1])
print(data[1].x)  # one hot encoded btw
# Gather some statistics about the first graph.
print(f'Number of nodes: {data[0].num_nodes}')
print(f'Number of edges: {data[0].num_edges}')
print(f'Average node degree: {data[0].num_edges / data[0].num_nodes:.2f}')
print(f'Has isolated nodes: {data[0].has_isolated_nodes()}')
print(f'Has self-loops: {data[0].has_self_loops()}')
print(f'Is undirected: {data[0].is_undirected()}')

# split data
n = len(data)
train = data[0:int(n*0.8)]  # 80%
test = data[int(n*0.8):]  # 20%
print(f'Train data len:{len(train)}\nTest data len:{len(test)}')

train_loader = DataLoader(train, batch_size=1000, shuffle=True)
test_loader = DataLoader(test, batch_size=1000, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 4)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64, dropout=0.6)


# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        true = data.y.argmax(dim=1)
        correct += int((pred == true).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


previousAcc = 1e-6
n = 0
allAccuracies = []
for epoch in range(1, 999):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    allAccuracies.append(test_acc)

    if test_acc < previousAcc:
        # count
        n += 1

    else:
        n = 0  # reset
        previousAcc = test_acc

    if n == 50:
        break

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Streak: {n}')

plt.plot(allAccuracies)
plt.title('Test accuracy GCN')
plt.ylabel('Accuracy')
plt.xlabel('Episode')
plt.show()
