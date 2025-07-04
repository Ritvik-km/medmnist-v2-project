import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from mmxp.data.pathmnist_loader import _get_dataset 
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import os

# ---- Config ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9
BATCH_SIZE = 128

base_dir    = os.path.dirname(os.path.abspath(__file__))       # …/analysis
repo_root   = os.path.abspath(os.path.join(base_dir, "..", ".."))
results_dir = os.path.join(repo_root, "results")
plots_dir   = os.path.join(results_dir, "plots")
chkpt_dir   = os.path.join(results_dir, "checkpoints")
os.makedirs(plots_dir, exist_ok=True)

model_path  = os.path.join(chkpt_dir, "resnet18_pathmnist.pth")


test_dataset = _get_dataset("test", download=True)  # same transform if you like
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Load Model ----
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---- Collect Predictions ----
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.squeeze().numpy())
        all_images.extend(images.cpu())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ---- Confusion Matrix ----
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - ResNet18 on PathMNIST")
plt.tight_layout()
save_path_cm = os.path.join(plots_dir, "confusion_matrix_pathmnist.png")
plt.savefig(save_path_cm, bbox_inches='tight')
print(f"✅ Saved confusion matrix to {save_path_cm}")
plt.show()

# ---- Misclassified Images ----
mis_idx = np.where(all_preds != all_labels)[0]
print(f"Total misclassified images: {len(mis_idx)}")

# Plot 10 misclassified examples
plt.figure(figsize=(15, 4))
for i, idx in enumerate(mis_idx[:10]):
    img = all_images[idx].permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
    true_label = all_labels[idx]
    pred_label = all_preds[idx]
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}, Pred: {pred_label}")
    plt.axis('off')

plt.suptitle("Examples of Misclassified Images")
plt.tight_layout()
save_path_mis = os.path.join(plots_dir, "misclassified_examples_pathmnist.png")
plt.savefig(save_path_mis, bbox_inches='tight')
print(f"✅ Saved misclassified examples to {save_path_mis}")
plt.show()
