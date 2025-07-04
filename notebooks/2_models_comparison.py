import os
import pandas as pd
import torch
from mmxp.models.mobilenet_v1 import MobileNetV1
from torchvision.models import resnet18
import torch.nn as nn

# --- Paths ---
results_dir = "results"
chkpt_dir = os.path.join(results_dir, "checkpoints")
log_dir = os.path.join(results_dir, "logs")

models_info = {
    "ResNet-18": {
        "log": os.path.join(log_dir, "pathmnist_resnet18_log.csv"),
        "ckpt": os.path.join(chkpt_dir, "resnet18_pathmnist.pth"),
        "build_fn": lambda: resnet18(weights=None),
        "num_classes": 9
    },
    "MobileNetV1": {
        "log": os.path.join(log_dir, "pathmnist_mobilev1_log.csv"),
        "ckpt": os.path.join(chkpt_dir, "mobilev1_pathmnist.pth"),
        "build_fn": lambda: MobileNetV1(num_classes=9),
        "num_classes": 9
    }
}

# --- Output Storage ---
rows = []

# --- Iterate over models ---
for name, info in models_info.items():
    # Build model
    model = info["build_fn"]()
    if name == "ResNet-18":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, info["num_classes"])

    model.load_state_dict(torch.load(info["ckpt"], map_location="cpu"))
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Read log file (last epoch)
    df = pd.read_csv(info["log"])
    last = df.iloc[-2] if df.iloc[-1]["epoch"] == "final" else df.iloc[-1]
    val_acc = float(last["val_acc"])
    val_auc = float(last["val_auc"])

    rows.append({
        "Model": name,
        "Params": f"{num_params:,}",
        "Val Acc": f"{val_acc:.4f}",
        "Val AUC": f"{val_auc:.4f}"
    })

# --- Display table ---
df_summary = pd.DataFrame(rows)
print(df_summary.to_markdown(index=False))

