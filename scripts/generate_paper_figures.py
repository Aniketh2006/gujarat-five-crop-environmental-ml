from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
RESULTS = ROOT / "outputs"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

def save(fig, filename):
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("CREATED:", path)

# FIGURE 1: Cultivation AUC (LODO)
df = pd.read_csv(
    RESULTS / "stage3_model_comparison.csv"
)

cult = df[df["task"] == "cultivation"].copy()

order = [
    ("logistic", "Logistic"),
    ("random_forest", "Random forest"),
    ("xgboost", "XGBoost"),
    ("crop_only_baseline", "Crop-only baseline")
]

values = []
labels = []

for model, label in order:
    row = cult[cult["model"] == model]
    assert len(row) == 1, f"Missing model: {model}"
    values.append(float(row.iloc[0]["AUC"]))
    labels.append(label)

fig, ax = plt.subplots(figsize=(5.5, 3.0))
ax.barh(labels, values)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Leave-one-district-out AUC")
ax.set_title("Cultivation prediction: model comparison")

for i, v in enumerate(values):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center",
            fontsize=8)

fig.tight_layout()
save(fig, "cultivation_lodo.pdf")

# FIGURE 2: Yield prediction R2
yield_df = df[df["task"] == "log_yield"].copy()

models = [
    "crop_year_baseline",
    "ridge",
    "random_forest",
    "xgboost"
]

names = ["Crop-year", "Ridge", "RF", "XGBoost"]

preferred = []
strict = []

for model in models:
    for subset, destination in [
        ("preferred", preferred),
        ("strict", strict)
    ]:
        row = yield_df[
            (yield_df["model"] == model) &
            (yield_df["subset"] == subset)
        ]
        assert len(row) == 1, (subset, model)
        destination.append(float(row.iloc[0]["R2"]))

x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.bar(x - width/2, preferred, width, label="Preferred")
ax.bar(x + width/2, strict, width, label="Strict")

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Pooled out-of-fold log-yield R²")
ax.set_title("Yield model comparison")
ax.legend(frameon=False)

fig.tight_layout()
save(fig, "yield_models.pdf")

# FIGURE 3: Existing SHAP result
src = RESULTS / "stage3_SHAP_descriptive.png"
assert src.exists(), f"Missing {src}"

shutil.copy2(src, OUT / "shap_descriptive.png")
print("COPIED:", OUT / "shap_descriptive.png")

# FIGURE 4: Spatial-block Kriging comparison
spatial = pd.read_csv(
    RESULTS / "spatial_block_kriging" /
    "spatial_block_kriging_metrics.csv"
)

models = [
    ("Mean baseline", "Mean baseline"),
    ("IDW", "IDW"),
    ("Ordinary Kriging", "Ordinary Kriging")
]

rmse = []
names = []

for model, label in models:
    row = spatial[spatial["model"] == model]
    assert len(row) == 1, f"Missing {model}"
    assert row.iloc[0]["status"] == "COMPLETE"
    rmse.append(float(row.iloc[0]["RMSE"]))
    names.append(label)

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(names, rmse)

ax.set_ylabel("RMSE (district cultivation rate)")
ax.set_title("Five-block spatial interpolation")
ax.set_ylim(0, max(rmse) * 1.2)

for i, v in enumerate(rmse):
    ax.text(i, v + 0.005, f"{v:.3f}",
            ha="center", fontsize=8)

fig.tight_layout()
save(fig, "spatial_kriging.pdf")

# FIGURE 5: Existing Stage 4 map
src = RESULTS / "stage4_district_map.png"
assert src.exists(), f"Missing {src}"

shutil.copy2(src, OUT / "district_cultivation_map.png")
print("COPIED:", OUT / "district_cultivation_map.png")

print("\nPASS: All five figure assets generated.")
