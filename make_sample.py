import os, pandas as pd
src = r"C:\Users\Abhinav\CICIoMT2024"
dst = "data_sample"
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    if f.endswith(".csv") and "(1)" not in f:
        pd.read_csv(os.path.join(src, f), nrows=2000).to_csv(os.path.join(dst, f), index=False)
        print("saved", f)
