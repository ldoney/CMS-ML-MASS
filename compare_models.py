import os
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryROC, BinaryAUROC
import torch
import uproot
import json

def combine(d, generate_jets=True):
  return torch.cat(tuple([d[k] for k in d])) if generate_jets else d.get("muons")

if __name__ == "__main__":
  RUNS_DIR = "gnn/gnn_outs/model_output_dir"
  runs = ["just_gat", "just_gcn", "mixed_mu_gat", "mixed_mu_gcn"]
  #for n in [100, 1000, 10000, 100000, 1000000]:
  #  runs.append(str(n) + "_normalize")
  #  runs.append(str(n) + "_no_normalize")

  generate_jets = True
  best_gnn = None
  best_gnn_line = None
  ax = plt.figure()
  for k in runs:
    run_root = RUNS_DIR + "/" + k
    model_fname = run_root + "/model"
    model_json_data = run_root + "/json_data.json"
    with open(model_json_data, 'r') as file:
      model_features = json.load(file)

    data = torch.load(run_root + "/training_data.pt")
    out = torch.load(run_root + "/training_pred.pt")
    with torch.no_grad():
      metric = BinaryROC()#, pos_label="Signal")
      au_metric = BinaryAUROC()
      if generate_jets:
        mask = combine(data.train_mask_dict, generate_jets=generate_jets)
        pred_masked = [v[1] for v in out[mask].cpu()]
        pred_masked = torch.tensor(pred_masked).double()
        true_masked = combine(data.y_dict, generate_jets=generate_jets)[mask].clone().detach().cpu()
        fpr, tpr, _ = metric(pred_masked, true_masked)
        auroc = au_metric(pred_masked, true_masked)
      else:
        pred_masked = [v[1] for v in out[data.test_mask].cpu()]
        pred_masked = torch.tensor(pred_masked).double()
        true_masked = data.y[data.test_mask].clone().detach().cpu()
        fpr, tpr, _ = metric(pred_masked, true_masked)
        auroc = au_metric(pred_masked, true_masked)

    print(f"{k} has auroc {auroc}")
    line = None
    if "no_" in k:
      line, = plt.plot(tpr, 1-fpr, linestyle="dashed", label=f"GNN {k} ({auroc:3.4f})")
    else:
      line, = plt.plot(tpr, 1-fpr, label=f"GNN {k} ({auroc:3.4f})")

    if best_gnn == None:
      best_gnn = (k, auroc)
      best_gnn_line = line
    else:
      best_gnn_line = line if auroc > best_gnn[1] else best_gnn_line
      best_gnn = (k, auroc) if auroc > best_gnn[1] else best_gnn

  best_bdtg = None
  best_bdtg_line = None
  best_dnn = None
  best_dnn_line = None

#  # Root DNN/BDTG
#  ROOT_DIR = "bdtg_dnn/mass_output_dir"
#  # What I did before is do a bunch of runs... for this I'm just gonna take the first
#  runs_to_process = ["187-7-18-MASS-DNN/Run-0", "187-9-18-MASS-DNN/Run-0"]
#  runs_to_process = [ROOT_DIR + "/" + r for r in runs_to_process]
#
#  for run in runs_to_process:
#    if "Run-" in run:
#      name = run.split("/")[-2][0:-9] + "-" + run.split("/")[-1][-1]
#    else:
#      name = run.split("/")[-1][0:-9]
#    with uproot.open(run + "/TMVA.root") as file:
#      try:
#        bdt_dir = file["dataset"]["Method_BDT"]["BDTG"]
#        bdt_roc = bdt_dir["MVA_BDTG_rejBvsS"]
#        x, y = bdt_roc.to_numpy()
#        auroc = np.trapz(x,y[1:])
#        print(f"{name} BDTG has auroc {auroc}")
#        # This is probably bad, for some reason y has 1 more value than x... so I just chop off the first, but that's probably unsafe
#        line, = plt.plot(x, y[1:], label=f"BDTG {name}")
#        if best_bdtg == None:
#          best_bdtg = (run, auroc)
#          best_bdtg_line = line
#        else:
#          best_bdtg_line = line    if auroc > best_bdtg[1] else best_bdtg_line
#          best_bdtg = (run, auroc) if auroc > best_bdtg[1] else best_bdtg
#      except Exception as exc:
#        #print(exc)
#        print(f"No DNN found for run {name}!")
#      try:
#        dnn_dir = file["dataset"]["Method_DL"]["TMVA_DNN_GPU"]
#        dnn_roc = dnn_dir["MVA_TMVA_DNN_GPU_rejBvsS"]
#        x, y = dnn_roc.to_numpy()
#        auroc = np.trapz(x,y[1:])
#        print(f"{name} DNN has auroc {auroc}")
#        line, = plt.plot(x, y[1:], label=f"DNN {name}")
#        if best_dnn == None:
#          best_dnn = (run, auroc)
#          best_dnn_line = line
#        else:
#          best_dnn_line = line    if auroc > best_dnn[1] else best_dnn_line
#          best_dnn = (run, auroc) if auroc > best_dnn[1] else best_dnn
#      except Exception as exc:
#        #print(exc)
#        print(f"No DNN found for run {name}!")

  if best_gnn_line != None:
    best_gnn_line.set_linewidth(4)
    print(f"Best GNN: {best_gnn[0]} with auroc {best_gnn[1].item()}")
  if best_dnn_line != None:
    best_dnn_line.set_linewidth(4)
    if "Run-" in best_dnn[0]:
      name = best_dnn[0].split("/")[-2][0:-9] + "-" + best_dnn[0].split("/")[-1][-1]
    else:
      name = best_dnn[0].split("/")[-1][0:-9]
    print(f"Best DNN: {name} with auroc {best_dnn[1].item()}")
  if best_bdtg_line != None:
    best_bdtg_line.set_linewidth(4)
    if "Run-" in best_bdtg[0]:
      name = best_bdtg[0].split("/")[-2][0:-9] + "-" + best_bdtg[0].split("/")[-1][-1]
    else:
      name = best_bdtg[0].split("/")[-1][0:-9]
    print(f"Best BDTG: {name} with auroc {best_bdtg[1].item()}")

  plt.title("All ROC curves plotted together")
  plt.xlabel('True Positive Rate')
  plt.ylabel('False Positive Rate' )
  ax.legend(loc=3)
  plt.show()

  plt.savefig(f"combined_roc_curve.png")
