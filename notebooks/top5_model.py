import os
import pandas as pd
from glob import glob

log_dir = "results/logs/"
log_files = sorted(glob(os.path.join(log_dir, "*.csv")))

results = []

for f in log_files:
    df = pd.read_csv(f)
    if df.empty:
        continue
    last = df.iloc[-2] if df.iloc[-1]["epoch"] == "final" else df.iloc[-1]
    val_acc = float(last["val_acc"])
    val_auc = float(last["val_auc"])
    model_name = os.path.basename(f).replace("pathmnist_", "").replace("_log.csv", "")
    results.append((model_name, f, val_acc, val_auc))


# Sort by validation accuracy
results.sort(key=lambda x: x[2], reverse=True)

print("🏆 Top-5 Models by Validation Accuracy:\n")
for i, (model, f, acc, auc) in enumerate(results[:5], 1):
    print(f"{i}. {model}")
    print(f"   Val Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print(f"   Model path: results/checkpoints/{model}_pathmnist.pth\n")

