import argparse, pickle, warnings
import pandas as pd
warnings.filterwarnings("ignore")

from pipeline import load_data, train_models

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="interns_dataset.csv", help="Path to your CSV")
parser.add_argument("--out",  default="pretrained_model.pkl", help="Output model file")
args = parser.parse_args()

print(f"Loading data from: {args.data}")
df = load_data(args.data)
print(f"Rows: {len(df):,}  |  Training now (this takes ~60s)…\n")

models, best_model, best_name, results, fi, X_test, y_test, X_train, y_train = train_models(df)

print("=" * 50)
print(f"  Best model : {best_name}")
print(f"  R²         : {results[best_name]['r2']}")
print(f"  MAE        : {results[best_name]['mae']}")
print(f"  CV R²      : {results[best_name]['cv_r2']}")
print("=" * 50)

bundle = dict(best_model=best_model, best_name=best_name,
              results=results, fi=fi,
              X_test=X_test, y_test=y_test,
              X_train=X_train, y_train=y_train)

with open(args.out, "wb") as f:
    pickle.dump(bundle, f)

import os
size = round(os.path.getsize(args.out) / 1024 / 1024, 1)
print(f"\nSaved to {args.out}  ({size} MB)")
print("Run  streamlit run app.py  to launch the dashboard.")