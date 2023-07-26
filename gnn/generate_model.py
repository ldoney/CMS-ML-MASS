import os
import sys
sys.path.insert(1, "lib")
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
from torch_geometric.datasets import FakeHeteroDataset


from torch_geometric.nn import to_hetero

import json
from models import GCN
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib
import argparse
from torchmetrics.classification import BinaryROC, BinaryAUROC

from torch_geometric.nn import to_hetero

import shutil
from common import query_yes_no, query_count, create_graph
import torch.nn as nn
from datetime import datetime
from generate_csvs import generate_csv_data
from tqdm import tqdm, trange
from inputimeout import inputimeout, TimeoutOccurred

def combine(d, generate_jets=True):
  return torch.cat(tuple([d[k] for k in d])) if generate_jets else d.get("muons")

def visualize_classification_result(model, graph, generate_jets=True):
  model.eval()
  pred = model(graph.x_dict, graph.edge_index_dict, generate_jets=generate_jets).argmax(dim=1)
  m_masses = graph["muons"].x[graph["muons"].test_mask][:,4].cpu().numpy()
  if generate_jets:
    j_masses = graph["jets"].x[graph["jets"].test_mask][:,1].cpu().numpy()
  corrects = (pred[combine(graph.test_mask_dict, generate_jets=generate_jets)] ==
              combine(graph.y_dict, generate_jets=generate_jets)[
                combine(graph.test_mask_dict, generate_jets=generate_jets)]).cpu().numpy().astype(int)
  test_index = np.arange(len(graph["muons"].x) + (len(graph["jets"].x) if generate_jets else 0))[
    combine(graph.test_mask_dict, generate_jets=generate_jets).cpu().numpy()]
  g, y_m, y_j = convert_to_networkx(graph.cpu(), generate_jets=generate_jets)
  g_test = g.subgraph(test_index)

  plot_graph(g_test,
             [("green" if v == 1 else "red") if i < m_masses.size else
              ("lightgreen" if v == 1 else "orange") for i, v in enumerate(corrects)],
             "classification_result",
             lerp(5,30, np.concatenate((
               (m_masses+1)/2,
               (j_masses+1)/2
             )) if generate_jets else (m_masses + 1)/2))

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200, generate_jets=True):
  for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(graph.x_dict, graph.edge_index_dict, generate_jets=generate_jets)
    mask = combine(graph.train_mask_dict, generate_jets=generate_jets)
    #print(graph["muons"])
    loss = criterion(out[mask], combine(graph.y_dict, generate_jets=generate_jets)[mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    acc = eval_node_classifier(model, graph, combine(graph.val_mask_dict, generate_jets=generate_jets), generate_jets=generate_jets)[0]

    if epoch % 10 == 0:
      print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')
  return model

# Lerp between a and b
def lerp(a, b, t):
  return (1 - t) * a + t * b

def eval_node_classifier(model, graph, mask, generate_jets=True):
  model.eval()
  out = model(graph.x_dict, graph.edge_index_dict, generate_jets=generate_jets)
  pred = out.argmax(dim=1)
  correct = (pred[mask] == combine(graph.y_dict, generate_jets=generate_jets)[mask]).sum()
  acc = int(correct) / int(mask.sum())

  return acc, out

def convert_to_networkx(graph, n_sample=None, generate_jets = False):
  g = to_networkx(graph.to_homogeneous())

  y_m = graph["muons"].y.numpy()
  y_j = graph["jets"].y.numpy() if generate_jets else None

  if n_sample is not None:
    sampled_nodes = random.sample(g.nodes, n_sample)
    g = g.subgraph(sampled_nodes)
    y_m = y_m[sampled_nodes]
    y_j = y_j[sampled_nodes] if generate_jets else None

    return g, y_m, y_j
  return g, y_m, y_j


def plot_graph(g, y, fname, node_size=30, pos=None):
  plt.figure(figsize=(9, 7))
  nx.draw_spring(g, node_size=node_size, arrows=False, node_color=y)
  if bool(os.environ.get('DISPLAY', None)):
    plt.show()
  plt.savefig(f"{OUTPUT_DIR}/{fname}.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog="CMS GNN Runner",
    description="Given a dataset, generates a GNN model to analyze it",
    epilog="Written by Lincoln Doney, 2023"
  )

  parser.add_argument("modeldir", nargs='?', default=None, help="Directory to store output")
  parser.add_argument("-s", "--size", type=int, nargs='?', default=None, required=False, help="Amount of events to take from dataset. If generate-csvs is set to false, this cannot be set")
  parser.add_argument("-n", "--name", type=str, nargs='?', default=None, required=False, help="Name of this particular run, for access later")
  parser.add_argument("-t", "--test", type=float, nargs='?', default=0.8, required=False, help="Percent of events to use for test")
  parser.add_argument("-v", "--validation", type=float, nargs='?', default=0.05, required=False, help="Percent of events to use for validation")
  parser.add_argument('--csv-dir', type=str, nargs='?', default=None, required=False, dest="csv_dir", help="If generate csvs is set to false, where to get csv files from")
  parser.add_argument("--muons-keys", type=str, nargs='*', default=None, required=False, dest="muons_keys", help="Keys in ROOT tree to use for muons. Should be formatted as 'muons.KEY' or 'muPairs.KEY', so mass is 'muPairs.mass'")
  parser.add_argument("--jets-keys", type=str, nargs='*', default=None, required=False, dest="jets_keys", help="Keys in ROOT tree to use for jets. Should be formatted as 'jets.KEY' or 'jetPairs.KEY', so mass is 'jetPairs.mass'")

  parser.add_argument('--plot-roc', action=argparse.BooleanOptionalAction, dest="plot_roc", default=True, help="Whether or not to plot ROC curve when complete")
  parser.add_argument('--generate-csvs', action=argparse.BooleanOptionalAction, dest="generate_csvs", default=True, help="Generate CSV files. If set to false, existing csv directory is required through --csv_dir")
  parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, dest="normalize", default=True, help="Whether to normalize data or not")
  parser.add_argument('--jets', action=argparse.BooleanOptionalAction, dest="generate_jets", default=True, help="Whether to include jets or not")
  #parser.add_argument("-v", "--verbose", action='store_const', const=True)
  parser.add_argument("-V", "--visualize", action='store_const', const=True, default=False, required=False, help="Whether to visualize graphs or not")
  args = parser.parse_args()
  assert(not (args.generate_csvs == False and args.csv_dir == None))
  assert(not (args.generate_csvs == False and args.size != None))
  assert(not (args.generate_csvs == True and args.csv_dir != None))
  assert(not (args.generate_jets == False and args.jets_keys != None))
  assert(args.test + args.validation < 1)

  if args.muons_keys == None:
    args.muons_keys = ["muons.pt", "muons.charge", "muons.eta", "muons.phi", "muPairs.mass", "muPairs.pt", "muPairs.eta"]
  if args.jets_keys == None:
    args.jets_keys = ["jets.pt", "jets.mass", "jets.charge", "jets.eta", "jets.phi"]

  if not args.generate_csvs:
    csv_dir = args.csv_dir
    print(csv_dir)
    assert(os.path.exists(csv_dir))
    if csv_dir[-1] == "/":
      csv_dir = csv_dir[0:len(csv_dir) - 1]
    if os.path.exists(csv_dir + "/jet_members.csv"):
      args.jets = True
    args.jets = False

    args.csv_dir = csv_dir


  if args.size == None:
    args.size = 1000

  if args.modeldir == None:
    if args.name == None:
      now = datetime.now()
      current_time = now.strftime("%H_%M_%S")
      print("Current Time =", current_time)
      name = current_time
    else:
      name = args.name[0]

    OUTPUT_DIR_ALL = "gnn_outs/model_output_dir"
    OUTPUT_DIR = f"{OUTPUT_DIR_ALL}/{name}"
  else:
    if args.modeldir[-1] == "/":
      OUTPUT_DIR = args.modeldir[0:(len(args.modeldir) - 1)]
    else:
      OUTPUT_DIR = args.modeldir

  if os.path.exists(OUTPUT_DIR):
    if query_yes_no("Path exists! Remove?", default="yes"):
      shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

  os.mkdir(OUTPUT_DIR)

  if args.generate_csvs:
    print("Generating input data...")
    generate_csv_data(output_dir = OUTPUT_DIR, takeonly = args.size, normalize = args.normalize, generate_jets=args.generate_jets, jets_keys = args.jets_keys, muons_keys = args.muons_keys)

  data = create_graph(directory = OUTPUT_DIR, generate_jets=args.generate_jets, csv_dir=OUTPUT_DIR if args.csv_dir == None else args.csv_dir, jets_keys = args.jets_keys, muons_keys = args.muons_keys)

  # Generate test/validate/train masks
  split = T.RandomNodeSplit(num_val=args.validation, num_test=args.test, key="x")
  data = split(data)

  if args.visualize:
    # Visualize graph
    print("blue (0): background")
    print("red (1): signal")
    print("lightblue (0): background, jet")
    print("orange (1): signal, jet")

    g, y_m, y_j = convert_to_networkx(data, generate_jets=args.generate_jets)
    # This is a bad case of hard-coding... for my run right now, muPairs.mass is in index 4, so I'm using that
    m_masses = data["muons"].x[:,4].numpy()
    if args.generate_jets:
      j_masses = data["jets"].x[:,1].numpy()
    plot_graph(g,
               ["blue" if v == 0 else "red" for v in y_m] +
               (["lightblue" if v == 0 else "orange" for v in y_j] if args.generate_jets else []),
               "input_graph",
               node_size=lerp(5, 30,
                              np.concatenate((
                                (m_masses+1)/2,
                                (j_masses + 1)/2)) if args.generate_jets else (m_masses + 1)/2))

  m_num_classes = torch.unique(data["muons"].y).size()[0]
  m_num_features = data["muons"].x.size()[1]

  print(f"Number of muon classes: {m_num_classes}")
  print(f"Number of muon features: {m_num_features}")

  if args.generate_jets:
    j_num_classes = torch.unique(data["jets"].y).size()[0]
    j_num_features = data["jets"].x.size()[1]

    print(f"Number of jets classes: {j_num_classes}")
    print(f"Number of jets features: {j_num_features}")

    assert(j_num_classes == m_num_classes)

  num_classes = m_num_classes

  # Ensure data is properly formatted
  data.validate(raise_on_error=True)

  # Transfer to GPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = data.to(device)
  gcn = GCN(num_classes, generate_jets = args.generate_jets).to(device)

  # Train
  optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
  criterion = nn.CrossEntropyLoss()
  gcn = train_node_classifier(gcn, data, optimizer_gcn, criterion, generate_jets = args.generate_jets)

  # Get how successful training was
  test_acc, out = eval_node_classifier(gcn, data, combine(data.test_mask_dict, generate_jets = args.generate_jets), generate_jets = args.generate_jets)

  torch.save(gcn.state_dict(), f"{OUTPUT_DIR}/model")
  torch.save(optimizer_gcn.state_dict(), f"{OUTPUT_DIR}/optimizer")

  with open(OUTPUT_DIR + '/json_data.json', 'w') as outfile:
    if args.generate_jets:
      json.dump({'num_classes': num_classes, 'm_num_features': m_num_features, 'j_num_features': j_num_features}, outfile)
    else:
      json.dump({'num_classes': num_classes, 'm_num_features': m_num_features}, outfile)


  print(f"Test Acc: {test_acc:.3f}")

  print(f"Plotting ROC curve...")
  with torch.no_grad():
    metric = BinaryROC()#, pos_label="Signal")
    au_metric = BinaryAUROC()
    pred_masked = [v[1] for v in out[combine(data.test_mask_dict, generate_jets=args.generate_jets)].cpu()]
    pred_masked = torch.tensor(pred_masked).double()
    true_masked = combine(data.y_dict, generate_jets=args.generate_jets)[combine(data.test_mask_dict, generate_jets=args.generate_jets)].clone().detach().cpu()
    fpr, tpr, _ = metric(pred_masked, true_masked)
    auroc = au_metric(pred_masked, true_masked)
    print(f"Area under ROC: {auroc}")

  os.mkdir(f"{OUTPUT_DIR}/roc_data")

  torch.save(out, f"{OUTPUT_DIR}/training_pred.pt")
  torch.save(data, f"{OUTPUT_DIR}/training_data.pt")
  np.save(f"{OUTPUT_DIR}/roc_data/tpr.npy", tpr)
  np.save(f"{OUTPUT_DIR}/roc_data/fpr.npy", fpr)

  if args.plot_roc:
    plt.title(f"ROC curve (area {auroc})")
    plt.plot(tpr, 1-fpr, marker='.')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate' )
    plt.show()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")

  print(f"Plot complete!")

  with open(OUTPUT_DIR + '/roc_area.json', 'w') as outfile:
    json.dump({'auroc': auroc.item()}, outfile)

  # Visualization
  if args.visualize:
    visualize_classification_result(gcn, data, generate_jets=args.generate_jets)
