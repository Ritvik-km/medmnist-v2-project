import os
import subprocess

DATASETS = [
        "pathmnist", "chestmnist", "dermamnist", "bloodmnist", "organcmnist", "organamnist", "organsmnist", "retinamnist", "breastmnist",     "pneumoniamnist", "tissuemnist", "octmnist"
    ]
MODELS = ["resnet18", "mobilenetv1"]
WIDTHS = [0.5, 0.75, 1.0]
LRS = [1e-3, 5e-4, 1e-4]

for dataset in DATASETS:
    for model in MODELS:
        if model == "mobilenetv1":
            for width in WIDTHS:
                for lr in LRS:
                    print(f"🚀 {model} | {dataset} | width={width}, lr={lr}")
                    cmd = f"python scripts/train.py --dataset {dataset} --model {model} --width {width} --lr {lr}"
                    subprocess.run(cmd, shell=True)
        else:
            for lr in LRS:
                print(f"🚀 {model} | {dataset} | lr={lr}")
                cmd = f"python scripts/train.py --dataset {dataset} --model {model}"
                subprocess.run(cmd, shell=True)
