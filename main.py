import argparse
import copy

from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from sklearn import metrics
from utils.hypergraph import Hypergraph
from gnn_model.model import HGFND, PropagationEncoder
from utils.data_loader import *
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='politifact',
                    choices=['politifact', 'gossipcop'])
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size for propagation encoding')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay penalty')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=200, help='epoch size')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffling training index')
parser.add_argument('--seed', type=int, default=777, help='random seed')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='data', feature="bert", empty=False, name=args.dataset, transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_train = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_train + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])

loader = DataLoader

all_loader = loader(dataset, batch_size=len(dataset), shuffle=False)
train_loader = loader(training_set, batch_size=args.batchSize, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batchSize, shuffle=False)
test_loader = loader(test_set, batch_size=args.batchSize, shuffle=False)

all_dataset = all_loader.dataset
train_dataset = train_loader.dataset
val_dataset = val_loader.dataset
test_dataset = test_loader.dataset

for data in all_loader:
    data = data.to(device)
    data_batch = data.batch
    data_edge_index = data.edge_index
    data_nodes = data.x
    data_labels = data.y

train_idx = train_dataset.indices
num_train = len(train_idx)
val_idx = val_dataset.indices
test_idx = test_dataset.indices
nodes_seq = np.arange(len(data_labels))

builder = Hypergraph(args)
hypergraph = builder.get_hyperedges()
alias_inputs, HT, node_masks = builder.get_adj_matrix(hypergraph, nodes_seq)

HT = torch.Tensor(np.array(HT)).float().to(device)

hypergraph_model = HGFND(args)
model = PropagationEncoder(args, hypergraph_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(train_idx):
    model.train()

    total_loss = 0
    for idx in range(0, num_train, args.batchSize):
        slices = train_idx[idx:min(idx + args.batchSize, num_train)]
        optimizer.zero_grad()
        out, _ = model(data_nodes, data_edge_index, HT, data_batch, slices)
        scores = model.compute_scores(out)
        labels = data_labels[slices]
        loss = F.nll_loss(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * len(slices)

    return total_loss / num_train


@torch.no_grad()
def test(test_idx, verbose=False):
    model.eval()

    num_samples = len(test_idx)
    test_preds = []
    out_log = []
    for idx in range(0, num_samples, args.batchSize):
        slices = test_idx[idx:min(idx + args.batchSize, num_samples)]
        output, _ = model(data_nodes, data_edge_index, HT, data_batch, slices)
        scores = model.compute_scores(output)
        pred = scores.argmax(dim=-1)
        test_preds += list(pred.detach().cpu().numpy())
        temp_labels = data_labels[slices]
        out_log.append([scores, temp_labels.view(-1, 1)])

    test_labels = data_labels[test_idx]
    test_labels = list(test_labels.detach().cpu().numpy())
    acc = metrics.accuracy_score(test_labels, test_preds)
    details = []
    if verbose:
        f1_macro, f1_micro, precision, recall = 0, 0, 0, 0
        f1_macro += metrics.f1_score(test_labels, test_preds, average='macro')
        f1_micro += metrics.f1_score(test_labels, test_preds, average='micro')
        precision += metrics.precision_score(test_labels, test_preds, zero_division=0)
        recall += metrics.recall_score(test_labels, test_preds, zero_division=0)
        details = [f1_macro, f1_micro, precision, recall]
    return acc, details


best_val_acc = 0
for epoch in range(1, args.epoch):
    loss = train(train_idx)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    val_acc, _ = test(val_idx, verbose=False)
    print(f'Val Accuracy: {val_acc:.4f}')
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        state_dict = copy.deepcopy(model.state_dict())

print("**Loading The Best model on Validation Dataset: ")
model.load_state_dict(state_dict)

acc, [f1_macro, f1_micro, precision, recall] = test(test_idx, verbose=True)
print("Test result: ", acc, f1_macro, f1_micro, precision, recall)
