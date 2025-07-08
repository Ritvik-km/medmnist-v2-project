import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from medmnist import INFO, Evaluator
from medmnist.dataset import PathMNIST
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import csv
from mmxp.data.loader import get_loaders
import argparse

# --- CLI Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "mobilenetv1"])
parser.add_argument('--width', type=float, default=1.0, help="Width multiplier (for MobileNetV1 only)")
parser.add_argument('--lr', type=float, default=None, help="Learning rate override")
parser.add_argument('--dataset', type=str, default="pathmnist",
                    choices=["pathmnist", "chestmnist", "dermamnist", "bloodmnist", "organcmnist", "organamnist", "organsmnist", "retinamnist", "breastmnist", "pneumoniamnist", "tissuemnist", "octmnist"], help="2D MedMNIST dataset flag")
args = parser.parse_args()


# Configuration
DATA_FLAG = args.dataset
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = args.lr if args.lr else 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9

info = INFO[DATA_FLAG]
task = info['task'] 

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.abspath(os.path.join(base_dir, "..", "results"))
chkpt_dir   = os.path.join(results_dir, "checkpoints")
log_dir     = os.path.join(results_dir, "logs")

os.makedirs(chkpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Load Data
train_loader, val_loader, test_loader = get_loaders(dataset_flag = DATA_FLAG, batch_size=BATCH_SIZE)

# Model
if args.model == "resnet18":
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model_name = "resnet18"
else:
    from mmxp.models.mobilenet_v1 import MobileNetV1
    model = MobileNetV1(num_classes=NUM_CLASSES, width_mult=args.width)
    model_name = f"mobilenetv1_w{args.width}_lr{args.lr}"

# log_file = os.path.join(log_dir, "pathmnist_resnet18_log.csv")
log_file = os.path.join(log_dir, f"{DATA_FLAG}_{model_name}_log.csv")

# Loss & Optimizer
if task == "multi-label":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate(model, loader, split_name):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            if task == "multi-label":
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_labels.extend(labels.squeeze().cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Metrics
    if task == "multi-label":
        acc = np.mean((all_preds == all_labels).all(axis=1))  # exact match ratio
        auc = roc_auc_score(all_labels, all_probs, average="macro")
    elif task == "binary":
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:  # multi-class
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    print(f"{split_name} Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['final', '', acc, auc])

    return acc, auc


# Initialize CSV log
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_acc', 'val_auc'])

# Training Loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images = images.to(DEVICE)
        if task == "multi-label":
            labels = labels.float().to(DEVICE)  # float for BCEWithLogits
        else:
            labels = labels.squeeze().long().to(DEVICE)  # long for CrossEntropy


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # Evaluate on val set
    val_acc, val_auc = evaluate(model, val_loader, split_name=f"Val (epoch {epoch+1})")

    # Log to file
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_loss, val_acc, val_auc])

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")


# Save model
# model_path = os.path.join(chkpt_dir, "resnet18_pathmnist.pth")
model_path = os.path.join(chkpt_dir, f"{model_name}_{DATA_FLAG}.pth")
torch.save(model.state_dict(), model_path)

# Evaluation
print("\nFinal Evaluation:")
evaluate(model, test_loader, "Test")
