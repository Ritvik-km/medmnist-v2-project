import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "..", "results")

# Load CSV
log_path = '../results/pathmnist_resnet18_log.csv'
df = pd.read_csv(log_path)

# Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(df['epoch'], df['train_loss'], marker='o')
axs[0].set_title('Training Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

axs[1].plot(df['epoch'], df['val_acc'], marker='o')
axs[1].set_title('Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')

axs[2].plot(df['epoch'], df['val_auc'], marker='o')
axs[2].set_title('Validation AUC')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('AUC')

plt.suptitle("ResNet-18 on PathMNIST: Training Curves")
plt.tight_layout()

save_path = os.path.join(results_dir, "training_curves_pathmnist.png")
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ Saved to {save_path}")
