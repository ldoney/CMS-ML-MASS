import os
for n in [1000000]: #[10, 100, 1000, 10000, 100000]:
  print(f"Running for n = {n}")
  print("Normalized run")
  os.system(f"python generate_model.py -s {n} -n {n}_normalize --normalize --no-plot-roc --generate-csvs")
  print("Non-normalized run")
  os.system(f"python generate_model.py -s {n} -n {n}_no_normalize --no-normalize --no-plot-roc --generate-csvs")
