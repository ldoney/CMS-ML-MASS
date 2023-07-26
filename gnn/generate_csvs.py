import uproot
import csv
from tqdm import tqdm
import numpy as np
import json

def generate_csv_data(output_dir=".", takeonly=1000000, generate_jets = False, normalize = True):
  print("Obtaining signal...")
  valid_typenames = ["float", "int64_t", "int32_t", "int16_t", "int8_t", "double", "uint8_t", "bool"]
  with uproot.open("signal_data.root") as sig_file:
    sig_tree = sig_file["dimuons"]["tree"]
    events_sig = sig_tree.arrays(entry_start=0, entry_stop=takeonly)
    events_keys = sig_tree.keys()
    types = sig_tree.typenames()

  if generate_jets:
    jets_events_keys_to_keep = ["jets.pt", "jets.mass", "jets.charge", "jets.eta", "jets.phi"]
    jets_events_branches_to_keep = list(set([s.split(".")[0] for s in jets_events_keys_to_keep]))
    jets_events_keys = [s for s in events_keys if types[s].split("[]")[0] in valid_typenames]
    jets_events_keys = [s.split("/")[-1] for s in jets_events_keys if any([b in s and b != s for b in jets_events_branches_to_keep])]
    jets_events_keys = [s for s in jets_events_keys if s in jets_events_keys_to_keep]

  events_keys_to_keep = ["muons.pt", "muons.charge", "muons.eta", "muons.phi", "muPairs.mass", "muPairs.pt", "muPairs.eta"]
  events_branches_to_keep = list(set([s.split(".")[0] for s in events_keys_to_keep]))
  events_keys = [s for s in events_keys if types[s].split("[]")[0] in valid_typenames]
  events_keys = [s.split("/")[-1] for s in events_keys if any([b in s and b != s for b in events_branches_to_keep])]
  events_keys = [s for s in events_keys if s in events_keys_to_keep]
  print(f"Using fields: {events_keys}")
  if generate_jets:
    print(f"Jets using fields: {jets_events_keys}")

  print("Obtained signal!")
  print("Obtaining background...")

  with uproot.open("background_data.root") as bg_file:
    bg_tree = bg_file["dimuons"]["tree"]
    events_bg = bg_tree.arrays(entry_start = 0, entry_stop=takeonly)

  print("Obtained background!")

  jets_x_combined = []

  x_combined = []
  m_m_edge_listing = []
  m_j_edge_listing = []
  j_m_edge_listing = []
  j_j_edge_listing = []

  print("Processing Signal Data...")
  for i, dR in tqdm(enumerate(events_sig["muPairs.dR"]), total=len(events_sig["muPairs.dR"])):
    if(len(events_sig["muons.pt"][i]) == 2):
      n = len(x_combined)

      new_arr = [n, "Signal"]
      for k in events_keys:
        e = events_sig[k][i]
        if len(e) > 0:
          new_arr.append(e[0])
        else:
          new_arr.append(e[0])
      x_combined.append(new_arr)

      new_arr = [n+1, "Signal"]
      for k in events_keys:
        e = events_sig[k][i]
        if len(e) > 1:
          new_arr.append(e[1])
        else:
          new_arr.append(e[0])
      x_combined.append(new_arr)

      m_m_edge_listing.append((n,n+1, dR[0]))
      m_m_edge_listing.append((n+1,n, dR[0]))

      n_jets = len(jets_x_combined)
      if generate_jets:
        n_jets_in_event = events_sig["nJets"][i]
        for j in range(0, n_jets_in_event):
          new_arr = [n_jets + j, "Signal"]
          for l in jets_events_keys:
            e = events_sig[l][i]
            new_arr.append(e[j])

          jets_x_combined.append(new_arr)
          m_j_edge_listing.append((n,     n_jets + j, dR[0]))
          m_j_edge_listing.append((n + 1, n_jets + j, dR[0]))

          j_m_edge_listing.append((n_jets + j, n,     dR[0]))
          j_m_edge_listing.append((n_jets + j, n + 1, dR[0]))
          for k in range(0, n_jets_in_event):
            if k != j:
              j_j_edge_listing.append((n_jets + j, n_jets + k, dR[0]))

      # TODO: Find something better than dR for this

  print("Processing Background Data...")
  for i, dR in tqdm(enumerate(events_bg["muPairs.dR"]), total=len(events_bg["muPairs.dR"])):
    if(len(events_sig["muons.pt"][i]) == 2):
      n = len(x_combined)

      new_arr = [n, "Background"]
      for k in events_keys:
        e = events_bg[k][i]
        if len(e) > 0:
          new_arr.append(e[0])
        else:
          new_arr.append(e[0])

      x_combined.append(new_arr)
      new_arr = [n+1, "Background"]
      for k in events_keys:
        e = events_bg[k][i]
        if len(e) > 1:
          new_arr.append(e[1])
        else:
          new_arr.append(e[0])

      x_combined.append(new_arr)

      m_m_edge_listing.append((n,n+1, dR[0]))
      m_m_edge_listing.append((n+1,n, dR[0]))

      n_jets = len(jets_x_combined)
      if generate_jets:
        n_jets_in_event = events_sig["nJets"][i]
        for j in range(0, n_jets_in_event):
          new_arr = [n_jets + j, "Background"]
          for l in jets_events_keys:
            e = events_sig[l][i]
            new_arr.append(e[j])
          jets_x_combined.append(new_arr)
          m_j_edge_listing.append((n,     n_jets + j, dR[0]))
          m_j_edge_listing.append((n + 1, n_jets + j, dR[0]))

          j_m_edge_listing.append((n_jets + j, n,     dR[0]))
          j_m_edge_listing.append((n_jets + j, n + 1, dR[0]))
          for k in range(0, n_jets_in_event):
            if k != j:
              j_j_edge_listing.append((n_jets + j, n_jets + k, dR[0]))

  if normalize:
    print("Normalizing...")
    x_combined = np.array([[col.astype(np.float64) if col_n >= 2 else col for col_n, col in enumerate(row)] for row_n, row in enumerate(x_combined)])

    # The first two values in each row is meta stuff (Id, signal/background)
    relevant_values = np.array(x_combined.transpose()[2:], dtype=np.float64)
    means = np.array([np.mean(c) for c in relevant_values], dtype=np.float64)
    sdevs = np.array([np.std(c) for c in relevant_values], dtype=np.float64)

    normalization_data = {}
    for i, k in enumerate(events_keys):
      normalization_data[k] = {"mean": means[i], "sdev": sdevs[i]}

    with open(output_dir + '/normalization_data.json', 'w') as outfile:
      json.dump(normalization_data, outfile)

    for i, row in tqdm(enumerate(x_combined)):
      for j, col in enumerate(row[2:]):
        x_combined[i][j + 2] = (col.astype(np.float64) - means[j].astype(np.float64))/sdevs[j].astype(np.float64)

    if generate_jets:
      jets_x_combined = np.array([[col.astype(np.float64) if col_n >= 2 else col for col_n, col in enumerate(row)] for row_n, row in enumerate(jets_x_combined)])

      # The first two values in each row is meta stuff (Id, signal/background)
      relevant_values = np.array(jets_x_combined.transpose()[2:], dtype=np.float64)
      means = np.array([np.mean(c) for c in relevant_values], dtype=np.float64)
      sdevs = np.array([np.std(c) for c in relevant_values], dtype=np.float64)

      normalization_data = {}
      for i, k in enumerate(jets_events_keys):
        normalization_data[k] = {"mean": means[i], "sdev": sdevs[i]}

      with open(output_dir + '/normalization_data_jets.json', 'w') as outfile:
        json.dump(normalization_data, outfile)

      for i, row in tqdm(enumerate(jets_x_combined)):
        for j, col in enumerate(row[2:]):
          jets_x_combined[i][j + 2] = (col.astype(np.float64) - means[j].astype(np.float64))/sdevs[j].astype(np.float64)

  print("Writing edge data to interactions.csv")
  f = open(output_dir + '/muon_muon_interactions.csv', 'w')
  writer = csv.writer(f)
  writer.writerow(["Src", "Dst", "Weight"])
  for src, dst, weight in tqdm(m_m_edge_listing):
    writer.writerow([src, dst, weight])
  f.close()

  print("Writing muons node data to muon_members.csv")
  f = open(output_dir + '/muon_members.csv', 'w')
  writer = csv.writer(f)
  writerow = ["Id", "SigBg"]
  for s in events_keys:
    writerow.append(s)

  writer.writerow(writerow)
  for arr in tqdm(x_combined):
    writer.writerow(arr)
  f.close()

  if generate_jets:
    print("Writing jets node data to jet_members.csv")
    f = open(output_dir + '/jet_members.csv', 'w')
    writer = csv.writer(f)
    writerow = ["Id", "SigBg"]
    for s in jets_events_keys:
      writerow.append(s)

    writer.writerow(writerow)
    for arr in tqdm(jets_x_combined):
      writer.writerow(arr)
    f.close()

    f = open(output_dir + '/muon_jet_interactions.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["Src", "Dst", "Weight"])
    for src, dst, weight in tqdm(m_j_edge_listing):
      writer.writerow([src, dst, weight])
    f.close()

    f = open(output_dir + '/jet_muon_interactions.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["Src", "Dst", "Weight"])
    for src, dst, weight in tqdm(j_m_edge_listing):
      writer.writerow([src, dst, weight])
    f.close()

    f = open(output_dir + '/jet_jet_interactions.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["Src", "Dst", "Weight"])
    for src, dst, weight in tqdm(j_j_edge_listing):
      writer.writerow([src, dst, weight])
    f.close()


if __name__ == "__main__":
  generate_csv_data(takeonly=10, normalize=True)
