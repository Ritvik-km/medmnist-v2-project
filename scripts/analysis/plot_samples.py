import matplotlib.pyplot as plt
from mmxp.data.pathmnist_loader import _get_dataset
import torchvision.transforms as transforms

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset (no normalization to keep original appearance)
dataset = _get_dataset("train", download=True)

# Track 1 image per class
class_to_image = {}
for img, label in dataset:
    label = int(label.item())
    if label not in class_to_image:
        class_to_image[label] = img
    if len(class_to_image) == 9:
        break

# Plot
fig, axs = plt.subplots(1, 9, figsize=(18, 2))
for idx, (label, img) in enumerate(sorted(class_to_image.items())):
    img = img.permute(1, 2, 0)  # CxHxW → HxWxC
    axs[idx].imshow(img)
    axs[idx].set_title(f"Class {label}")
    axs[idx].axis('off')

plt.suptitle("One Sample per Class from PathMNIST")
plt.tight_layout()
plt.show()
