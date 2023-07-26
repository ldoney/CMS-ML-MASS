import pandas as pd
from tqdm import tqdm, trange
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, HeteroData

def create_graph(directory="gnn_outs", generate_jets = False):
  # Collect data
  print("Reading csv files...")
  m_nodes_data = pd.read_csv(directory + "/muon_members.csv")
  if generate_jets:
    j_nodes_data = pd.read_csv(directory + "/jet_members.csv")
  m_m_edges_data = pd.read_csv(directory + "/muon_muon_interactions.csv")
  if generate_jets:
    j_m_edges_data = pd.read_csv(directory + "/jet_muon_interactions.csv")
    m_j_edges_data = pd.read_csv(directory + "/muon_jet_interactions.csv")
    j_j_edges_data = pd.read_csv(directory + "/jet_jet_interactions.csv")


  valid_dtypes = ['float64', 'bool', 'int64']
  m_properties = [s for s in m_nodes_data.keys() if s not in ["Id", "SigBg"]]
  m_properties = [p for p in m_properties if m_nodes_data[p].dtype in valid_dtypes]

  if generate_jets:
    j_properties = [s for s in j_nodes_data.keys() if s not in ["Id", "SigBg"]]
    j_properties = [p for p in j_properties if j_nodes_data[p].dtype in valid_dtypes]

  with tqdm(total=7 + (12 if generate_jets else 0)) as pbar:
    m_node_features = torch.from_numpy(m_nodes_data[m_properties].to_numpy()).double()
    if generate_jets:
      j_node_features = torch.from_numpy(j_nodes_data[j_properties].to_numpy()).double()

    pbar.update(1)
    m_categories = m_nodes_data["SigBg"].astype("category")
    m_codes = m_categories.cat.codes.to_numpy()

    m_codes.setflags(write=True)

    m_node_labels = torch.from_numpy(m_codes).long()

    if generate_jets:
      j_categories = j_nodes_data["SigBg"].astype("category")
      j_codes = j_categories.cat.codes.to_numpy()
      j_codes.setflags(write=True)
      j_node_labels = torch.from_numpy(j_codes).long()

    pbar.update(1)

    m_m_edge_features = torch.from_numpy(m_m_edges_data["Weight"].to_numpy()).double()
    pbar.update(1)

    m_m_edges_src = torch.from_numpy(m_m_edges_data["Src"].to_numpy()).long()
    pbar.update(1)

    m_m_edges_dst = torch.from_numpy(m_m_edges_data["Dst"].to_numpy()).long()
    pbar.update(1)

    m_m_edge_index = torch.tensor([m_m_edges_data["Src"], m_m_edges_data["Dst"]]).long()
    pbar.update(1)

  # Create data object
  print("Creating data object...")
  if generate_jets:
    m_j_edge_features = torch.from_numpy(m_j_edges_data["Weight"].to_numpy()).double()
    pbar.update(1)

    m_j_edges_src = torch.from_numpy(m_j_edges_data["Src"].to_numpy()).long()
    pbar.update(1)

    m_j_edges_dst = torch.from_numpy(m_j_edges_data["Dst"].to_numpy()).long()
    pbar.update(1)

    m_j_edge_index = torch.tensor([m_j_edges_data["Src"], m_j_edges_data["Dst"]]).long()
    pbar.update(1)

    j_m_edge_features = torch.from_numpy(j_m_edges_data["Weight"].to_numpy()).double()
    pbar.update(1)

    j_m_edges_src = torch.from_numpy(j_m_edges_data["Src"].to_numpy()).long()
    pbar.update(1)

    j_m_edges_dst = torch.from_numpy(j_m_edges_data["Dst"].to_numpy()).long()
    pbar.update(1)

    j_m_edge_index = torch.tensor([j_m_edges_data["Src"], j_m_edges_data["Dst"]]).long()
    pbar.update(1)

    j_j_edge_features = torch.from_numpy(j_j_edges_data["Weight"].to_numpy()).double()
    pbar.update(1)

    j_j_edges_src = torch.from_numpy(j_j_edges_data["Src"].to_numpy()).long()
    pbar.update(1)

    j_j_edges_dst = torch.from_numpy(j_j_edges_data["Dst"].to_numpy()).long()
    pbar.update(1)

    j_j_edge_index = torch.tensor([j_j_edges_data["Src"], j_j_edges_data["Dst"]]).long()
    pbar.update(1)

    data = HeteroData(
      {"muons": {
        "x": m_node_features.float(),
        "y": m_node_labels,
      },
      "jets": {
        "x": j_node_features.float(),
        "y": j_node_labels,
      },
      ("muons", "interacts", "muons"): {
        "edge_index": m_m_edge_index,
        "edge_weight": m_m_edge_features,
      },
      ("jets", "interacts", "muons"): {
        "edge_index": j_m_edge_index,
        "edge_weight": j_m_edge_features,
      },
      ("muons", "interacts", "jets"): {
        "edge_index": m_j_edge_index,
        "edge_weight": m_j_edge_features,
      },
      ("jets", "interacts", "jets"): {
        "edge_index": j_j_edge_index,
        "edge_weight": j_j_edge_features,
      }
    })
  else:
    data = HeteroData(
      {"muons": {
        "x": m_node_features.float(),
        "y": m_node_labels,
      },
      ("muons", "interacts", "muons"): {
        "edge_index": m_m_edge_index,
        "edge_weight": m_m_edge_features,
      }
    })
  pbar.update(1)
  return data

def query_yes_no(question, default="no"):
  valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
  if default is None:
    prompt = " [y/n] "
  elif default == "yes":
    prompt = " [Y/n] "
  elif default == "no":
    prompt = " [y/N] "
  else:
    raise ValueError("invalid default answer: '%s'" % default)

  while True:
    try:
      choice = inputimeout(prompt=question + prompt, timeout=3).lower()
    except:
      choice = default.lower()
    if default is not None and choice == "":
      return valid[default]
    elif choice in valid:
      return valid[choice]
    else:
      sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def query_count(question, default=0):
  while True:
    try:
      choice = inputimeout(prompt=question + prompt, timeout=3)
    except:
      choice = default
    if default is not None and choice == "":
      return default
    return int(choice)



