"""
Grad-CAM overlay grid for a PathMNIST checkpoint.
color scale (blue → green → yellow → red) used to visualize intensity

Title T:x P:y -
    T:x → True label (ground truth)
    P:y → Model prediction (class ID)

Example
-------
python scripts/interpret/grad_cam.py --checkpoint results/checkpoints/mobilenetv1_w0.5_lr0.001.pth --n 16 --cols 4

"""
import argparse, os, re, math, torch, torch.nn as nn, matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp   # pip install torchcam
from mmxp.data.loader import get_loaders

# ─────────────── CLI ────────────────
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--n", type=int, default=12, help="# validation images")
p.add_argument("--cols", type=int, default=4,  help="# subplots per row")
args = p.parse_args()

ckpt_path  = args.checkpoint
model_id   = os.path.basename(ckpt_path).replace(".pth", "")
is_mobile  = model_id.startswith("mobilenetv1")
width_mult = float(re.search(r"_w([0-9.]+)", model_id).group(1)) if is_mobile else None

DEVICE, NUM_CLASSES = ("cuda" if torch.cuda.is_available() else "cpu"), 9

# ─────── model factory ───────
if is_mobile:
    from mmxp.models.mobilenet_v1 import MobileNetV1
    model = MobileNetV1(num_classes=NUM_CLASSES, width_mult=width_mult)
    target_layer = model.model[-4]       # final depth-wise conv
else:
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.conv1, model.maxpool = nn.Conv2d(3,64,3,1,1,bias=False), nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    target_layer = model.layer4[-1].conv2

model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.to(DEVICE).eval()

cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)

# ─────── get a batch of validation imgs ───────
val_loader, _, _ = get_loaders(batch_size=args.n)
images, labels   = next(iter(val_loader))
images, labels   = images.to(DEVICE), labels.squeeze()

# forward **with gradients enabled**
logits = model(images)
preds  = logits.argmax(1)

# ─────── produce overlays ───────
plots_dir = os.path.join("results", "plots", model_id)
os.makedirs(plots_dir, exist_ok=True)

cols = args.cols
rows = math.ceil(args.n / cols)
plt.figure(figsize=(3*cols, 3*rows))

for idx in range(args.n):
    img      = images[idx:idx+1]
    class_id = preds[idx].item()
    cam_map  = cam_extractor(class_id, logits[idx:idx+1])[0].cpu()

    # normalise cam 0-1
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-5)

    # denormalise input (training used mean=.5, std=.5 → x*0.5+0.5)
    rgb = img.cpu().squeeze().permute(1,2,0).numpy()
    rgb = (rgb * 0.5 + 0.5).clip(0, 1)

    plt.subplot(rows, cols, idx+1)
    plt.imshow(rgb)
    plt.imshow(cam_map, cmap="jet", alpha=0.45)      # overlay
    plt.title(f"T:{labels[idx].item()} P:{class_id}", fontsize=8)
    plt.axis("off")

plt.tight_layout()
grid_path = os.path.join(plots_dir, "gradcam_grid.png")
plt.savefig(grid_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Grad-CAM grid saved → {grid_path}")
