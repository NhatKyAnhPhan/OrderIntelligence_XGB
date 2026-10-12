import argparse, csv, json, os

MODELS = [("RF", "random_forest"), ("GBT", "gbt"), ("XGBoost", "xgboost")]

p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--expect-rows", type=int, default=500_000_000)
a = p.parse_args()

header = ["Mo hinh", "MAE (USD)", "RMSE (USD)", "R2", "So dong", "Train (phut)", "Ghi chu"]
rows = []
for name, folder in MODELS:
    path = os.path.join(a.results, folder, "metrics.json")
    if not os.path.exists(path):
        rows.append([name, "-", "-", "-", "-", "-", "chua co ket qua"])
        continue
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    note = "" if m["rows"] == a.expect_rows else f"CHUA PHAI full ({m['rows']:,} dong)"
    rows.append([name, f"{m['mae']:.2f}", f"{m['rmse']:.2f}", f"{m['r2']:.4f}",
                 f"{m['rows']:,}", f"{m['train_time_sec'] / 60:.1f}", note])

widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

out = os.path.join(a.results, "results_table.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    csv.writer(f).writerows([header] + rows)
print("\nDa ghi", out)