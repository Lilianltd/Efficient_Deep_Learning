import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("model_accuracy_vs_parameters.csv")

# -----------------------------
# Create high-res figure
# -----------------------------
plt.figure(figsize=(12, 8))

plt.scatter(
    df["Trainable Parameters"],
    df["Accuracy (%)"]
)

# -----------------------------
# Add labels to each point
# -----------------------------
for i, row in df.iterrows():
    plt.text(
        row["Trainable Parameters"],
        row["Accuracy (%)"],
        row["Model"],
        fontsize=9,
        ha='right',
        va='bottom'
    )

# -----------------------------
# Add 90% accuracy line
# -----------------------------
plt.axhline(y=90, linestyle='--')

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Trainable Parameters")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy vs Number of Parameters")
plt.grid(True)

# -----------------------------
# Save high-quality image
# -----------------------------

# Best raster image (for papers/presentations)
plt.savefig(
    "accuracy_vs_parameters.png",
    dpi=1200,
    bbox_inches="tight"
)

# Best vector version (infinite resolution)
plt.savefig(
    "accuracy_vs_parameters.pdf",
    bbox_inches="tight"
)

plt.show()
