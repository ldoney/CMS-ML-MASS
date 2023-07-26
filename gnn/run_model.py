from os import walk
import sys
sys.path.insert(1, "lib")
from models import GCN
import torch
import argparse
from common import query_yes_no, create_graph
import torch.nn.functional as F

if __name__ == "__main__":
  OUTPUT_DIR = ""
  parser = argparse.ArgumentParser()

  parser.add_argument("modeldir", nargs='?', default=None)
  parser.add_argument("-c", "--classes", action='store_const', const=None)
  parser.add_argument("-f", "--features", action='store_const', const=None)
  args = parser.parse_args()

  if args.modeldir == None:
    OUTPUT_DIR_ALL = "gnn_outs/model_output_dir"

    dirs = []
    for (dirpath, dirnames, filenames) in walk(OUTPUT_DIR_ALL):
      dirs = dirnames
      break
    print("Found models for the following timestamps: ")
    [print(d, end="\t") for d in dirs]
    print("\n")
    to_get = input("Enter the name of the model to use: ")
    if to_get not in dirs:
      print("Invalid input!")
    OUTPUT_DIR = OUTPUT_DIR_ALL + "/" + to_get
  else:
    OUTPUT_DIR = args.modeldir

  if args.classes == None:
    num_classes = int(input("Enter the number of output classes (default 2): ") or 2)
  else:
    num_classes = args.classes

  if args.features == None:
    num_node_features = int(input("Enter the number of input features (default 7): ") or 7)
  else:
    num_classes = args.features

  model_fname = OUTPUT_DIR + "/model"
  #optimizer_fname = OUTPUT_DIR + "/optimizer"

  model = GCN(num_node_features, num_classes)
  model.load_state_dict(torch.load(model_fname))
  model.eval()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)

  correct = 0
  total_loss = 0
  n_graphs = 0
  with torch.no_grad():
    data = create_graph()

    out = model(data.to(device))

    total_loss += F.nll_loss(out, data.y).item()
    pred = out.max(1)[1]
    correct += pred.eq(data.y).sum().item()
    n_graphs += 1

  yay = correct# / (n_graphs * num_nodes)
  nay = total_loss
  print(f"Correct: {correct}/{len(data.x)}, {correct/len(data.x)}%")
  print(f"Loss: {total_loss}")
