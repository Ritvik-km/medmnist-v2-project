import subprocess
from itertools import product

# Define hyperparameter grid
widths = [0.5, 0.75, 1.0]
lrs = [1e-3, 5e-4, 1e-4]

# Loop through all combinations
for width, lr in product(widths, lrs):
    print(f"\n🚀 Training MobileNetV1 with width={width}, lr={lr}")
    
    command = [
        "python", "scripts/train.py",
        "--model", "mobilenetv1",
        "--width", str(width),
        "--lr", str(lr)
    ]
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"Failed: width={width}, lr={lr}")
    else:
        print(f"Completed: width={width}, lr={lr}")
