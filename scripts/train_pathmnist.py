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
from mmxp.data.pathmnist_loader import get_loaders


# Configuration
DATA_FLAG = 'pathmnist'
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9


base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.abspath(os.path.join(base_dir, "..", "results"))
chkpt_dir   = os.path.join(results_dir, "checkpoints")
log_dir     = os.path.join(results_dir, "logs")

os.makedirs(chkpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "pathmnist_resnet18_log.csv")

# Transforms (convert to 3 channels)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1 channel to 3
    transforms.Normalize(mean=[.5]*3, std=[.5]*3)
])

# Load Data
train_loader, val_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

# info = INFO[DATA_FLAG]
# DataClass = getattr(__import__('medmnist.dataset', fromlist=[info['python_class']]), info['python_class'])

# train_dataset = DataClass(split='train', transform=transform, download=True)
# val_dataset   = DataClass(split='val', transform=transform, download=True)
# test_dataset  = DataClass(split='test', transform=transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # Remove 7x7 maxpool layer for small inputs
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate(model, loader, split_name):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

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
        images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)

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
model_path = os.path.join(chkpt_dir, "resnet18_pathmnist.pth")
torch.save(model.state_dict(), model_path)

# Evaluation
print("\nFinal Evaluation:")
evaluate(model, test_loader, "Test")
