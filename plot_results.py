"""Ve 3 bieu do tu thu muc ket qua cua train_rf.py.

Chay:
  pip install matplotlib
  python plot_results.py --results results/rf_full_t30_d10
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--results", required=True, help="thu muc ket qua cua train_rf.py")
a = p.parse_args()


def path(name):
    return os.path.join(a.results, name)


# 1. Feature importance (top 10)
with open(path("feature_importance.json"), encoding="utf-8") as f:
    top = list(json.load(f).items())[:10]
names, values = zip(*top)
plt.figure(figsize=(8, 5))
plt.barh(names[::-1], values[::-1])
plt.xlabel("Importance")
plt.title("Top 10 feature importance - Random Forest")
plt.tight_layout()
plt.savefig(path("chart_feature_importance.png"), dpi=150)
plt.close()

# 2. Actual vs Predicted
df = pd.read_csv(path("actual_vs_predicted_sample.csv"))
actual, pred = df["total_price_usd"], df["prediction"]
plt.figure(figsize=(6, 6))
plt.scatter(actual, pred, s=2, alpha=0.3)
lim = [0, max(actual.max(), pred.max())]
plt.plot(lim, lim, "r--", linewidth=1)
plt.xlabel("Actual total_price_usd")
plt.ylabel("Predicted total_price_usd")
plt.title("Actual vs Predicted (mau ~50k dong)")
plt.tight_layout()
plt.savefig(path("chart_actual_vs_predicted.png"), dpi=150)
plt.close()

# 3. Phan bo sai so (Actual - Predicted)
plt.figure(figsize=(8, 5))
plt.hist(actual - pred, bins=100)
plt.xlabel("Sai so (Actual - Predicted, USD)")
plt.ylabel("So dong")
plt.title("Phan bo sai so du bao")
plt.tight_layout()
plt.savefig(path("chart_residuals.png"), dpi=150)
plt.close()

print("Da luu 3 bieu do vao", a.results)