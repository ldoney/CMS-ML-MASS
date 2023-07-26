import torch
#from torch_geometric.nn import GCNConv
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
import torch.nn.functional as F

class GCN(torch.nn.Module):
  def __init__(self, num_classes, hidden_channels=16, generate_jets=False, num_layers = 2):
    super().__init__()

    out_channels = num_classes
    self.generate_jets = generate_jets
    self.convs = torch.nn.ModuleList()
    for _ in range(num_layers):
      if generate_jets:
        conv = HeteroConv({
          ('muons', 'interacts', 'muons'): GATConv((-1,-1), hidden_channels),
          ('muons', 'interacts', 'jets'): SAGEConv((-1, -1), hidden_channels),
          ('jets', 'interacts', 'jets'): GCNConv(-1, hidden_channels),
        }, aggr='sum')
      else:
        conv = HeteroConv({
          ('muons', 'interacts', 'muons'): GCNConv(-1, hidden_channels),
        }, aggr='sum')

      self.convs.append(conv)
    self.lin = Linear(hidden_channels, out_channels)

    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x_dict, edge_index_dict, generate_jets=True):
    #x, edge_index = data["muons"].x, data["muons"].edge_index

    #x = x.float()
    for conv in self.convs:
      x_dict = conv(x_dict, edge_index_dict)
      x_dict = {key: x.float().relu() for key, x in x_dict.items()}
      out_t = torch.cat((x_dict["muons"], x_dict["jets"])) if generate_jets else x_dict["muons"]

      return self.sigmoid(self.lin(out_t))
