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

n_channels = info['n_channels']
n_classes = len(info['label'])
NUM_CLASSES = n_classes

is_multilabel = "multi-label" in task
is_binary     = ("binary"     in task) and not is_multilabel   # true for e.g. PneumoniaMNIST

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
    model = MobileNetV1(num_classes=NUM_CLASSES, input_channels=n_channels, width_mult=args.width)
    model_name = f"mobilenetv1_w{args.width}_lr{args.lr}"

# log_file = os.path.join(log_dir, "pathmnist_resnet18_log.csv")
log_file = os.path.join(log_dir, f"{DATA_FLAG}_{model_name}_log.csv")

# Loss & Optimizer
if is_multilabel:
    criterion = nn.BCEWithLogitsLoss()
elif is_binary:
    criterion = nn.BCEWithLogitsLoss()          # 1-logit or 2-logit is fine
else:                                           # multi-class
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model = model.to(DEVICE)

def evaluate(model, loader, split_name):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            if is_multilabel:
                probs = torch.sigmoid(outputs)          # [B, C]
                preds = (probs > 0.5).int()
                
                # Collect predictions and labels
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            
            elif is_binary:
                probs = torch.sigmoid(outputs).squeeze()        # [B]
                preds = (probs > 0.5).int()
                
                # Collect predictions and labels
                all_labels.extend(labels.view(-1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            
            else:  # multi-class
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Collect predictions and labels
                all_labels.extend(labels.squeeze().cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Compute metrics on all collected data
    if is_multilabel:
        acc = np.mean((all_preds == all_labels).all(axis=1))  # exact match ratio
        auc = roc_auc_score(all_labels, all_probs, average="macro")
    elif is_binary:
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
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
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        if images.size(0) != labels.size(0):
            print(f"⚠️ Batch {batch_idx}: size mismatch - images={images.size(0)}, labels={labels.size(0)}")
            continue  # Skip this problematic batch
            
        if images.size(0) == 0 or labels.size(0) == 0:
            print(f"⚠️ Batch {batch_idx}: empty batch - images={images.size(0)}, labels={labels.size(0)}")
            continue  # Skip empty batches
            
        images = images.to(DEVICE)
        if is_multilabel:
            labels = labels.float().to(DEVICE)
        elif is_binary:
            labels = labels.view(-1).float().to(DEVICE)
        else:
            if labels.dim() > 1:
                labels = labels.squeeze().long().to(DEVICE)
            else:
                labels = labels.long().to(DEVICE)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

        if images.size(0) != labels.size(0):
            print(f"⚠️ Batch {batch_idx}: post-processing size mismatch - images={images.shape}, labels={labels.shape}")
            continue

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
