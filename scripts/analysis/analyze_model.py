"""
Confusion-matrix + per-class metrics for PathMNIST checkpoints.

Usage
-----
python scripts/analysis/analyze_model.py --checkpoint results/checkpoints/mobilenetv1_w0.5_lr0.001.pth --batch-size 256 --n-mis 12

"""

import argparse, os, re, torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from mmxp.data.loader import get_loaders

# ───────────── CLI ─────────────
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--n-mis", type=int, default=10)
args = p.parse_args()

DEVICE, NUM_CLASSES = ("cuda" if torch.cuda.is_available() else "cpu"), 9
ckpt_path   = args.checkpoint
model_id    = os.path.basename(ckpt_path).replace(".pth", "")      # e.g. mobilenetv1_w0.5_lr0.001
is_mobile   = model_id.startswith("mobilenetv1")
width_mult  = float(re.search(r"_w([0-9.]+)", model_id).group(1)) if is_mobile else None

# ───────────── Model factory ─────────────
if is_mobile:
    from mmxp.models.mobilenet_v1 import MobileNetV1
    model = MobileNetV1(num_classes=NUM_CLASSES, width_mult=width_mult)
else:
    from torchvision.models import resnet18
    model           = resnet18(weights=None)
    model.conv1     = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool   = nn.Identity()
    model.fc        = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.to(DEVICE).eval()

# ───────────── Data ─────────────
_, _, test_loader = get_loaders(batch_size=args.batch_size)

all_logits, all_labels, all_imgs = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.squeeze().to(DEVICE)
        all_logits.append(model(x).cpu())
        all_labels.append(y.cpu())
        all_imgs.extend(x.cpu())

logits  = torch.cat(all_logits)
labels  = torch.cat(all_labels)
preds   = logits.argmax(1)

# ───────────── Metrics & Plots ─────────────
cm       = confusion_matrix(labels, preds)
reportDF = pd.DataFrame(classification_report(labels, preds, output_dict=True,
                                              zero_division=0)).T

plots_dir = os.path.join("results", "plots", model_id)
os.makedirs(plots_dir, exist_ok=True)

# Confusion matrix png
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix – {model_id}")
cm_png = os.path.join(plots_dir, "confusion_matrix.png")
plt.savefig(cm_png, bbox_inches="tight")
plt.close()

# Per-class CSV
csv_path = os.path.join(plots_dir, "per_class_metrics.csv")
reportDF.to_csv(csv_path, float_format="%.4f")

print(f"Saved  {cm_png}\n Saved  {csv_path}")

# Mis-classified thumbnails
mis_idx = (preds != labels).nonzero(as_tuple=True)[0][:args.n_mis]
if len(mis_idx):
    cols = min(args.n_mis, 5)
    rows = int(np.ceil(len(mis_idx)/cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i, idx in enumerate(mis_idx):
        img = all_imgs[idx].permute(1,2,0).numpy()
        img = (img*0.5+0.5).clip(0,1)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(f"T:{labels[idx].item()} P:{preds[idx].item()}", fontsize=8)
        plt.axis("off")
    mis_png = os.path.join(plots_dir, "misclassified_examples.png")
    plt.tight_layout()
    plt.savefig(mis_png, bbox_inches="tight")
    plt.close()
    print(f"Saved  {mis_png}")
