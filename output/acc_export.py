import torch
import pandas as pd
from models import *

# -----------------------------
# Helper: count trainable params
# -----------------------------
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------
# Models list
# -----------------------------
net_test = [
    VGG('VGG19'),
    ResNet18(),
    PreActResNet18(),
    GoogLeNet(),
    densenet_cifar(),
    DenseNet121(),
    ResNeXt29_2x64d(),
    MobileNet(),
    MobileNetV2(),
    DPN92(),
    SENet18(),
    ShuffleNetV2(1),
    EfficientNetB0(),
    RegNetX_200MF(),
    SimpleDLA()
]

names = [
    "VGG19",
    "ResNet18",
    "PreActResNet18",
    "GoogLeNet",
    "DenseNet_cifar",
    "DenseNet121",
    "ResNeXt29_2x64d",
    "MobileNet",
    "MobileNetV2",
    "DPN92",
    "SENet18",
    "ShuffleNetV2",
    "EfficientNetB0",
    "RegNetX_200MF",
    "SimpleDLA"
]

# -----------------------------
# Accuracy dictionary
# -----------------------------
accuracy = {
    "VGG19": 92.64,
    "ResNet18": 93.02,
    "ResNet50": 93.62,  # included for completeness
    "ResNet101": 93.75,
    "RegNetX_200MF": 94.24,
    "RegNetY_400MF": 94.29,
    "MobileNetV2": 94.43,
    "ResNeXt29(32x4d)": 94.73,
    "ResNeXt29_2x64d": 94.82,
    "SimpleDLA": 94.89,
    "DenseNet121": 95.04,
    "PreActResNet18": 95.11,
    "DPN92": 95.16,
    "DLA": 95.47
}

# -----------------------------
# Collect results
# -----------------------------
results = []

for name, model in zip(names, net_test):
    model.eval()

    params = count_trainable_params(model)
    acc = accuracy.get(name, None)

    results.append({
        "Model": name,
        "Accuracy (%)": acc,
        "Trainable Parameters": params
    })

# -----------------------------
# Save to CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("model_accuracy_vs_parameters.csv", index=False)

print("CSV saved as model_accuracy_vs_parameters.csv")
print(df)
