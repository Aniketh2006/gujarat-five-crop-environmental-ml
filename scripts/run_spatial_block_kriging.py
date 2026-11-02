from pathlib import Path
import json
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spatial_block_kriging"
OUT.mkdir(parents=True, exist_ok=True)

def normalize(s):
    return "".join(c for c in str(s).lower() if c.isalnum())

# Original Stage 4 observed district cultivation rates
observed = pd.read_csv(
    ROOT / "outputs/stage4_kriging_oof.csv"
)[["District", "observed"]]

# Reuse V9 geographic-block definitions
v9 = pd.read_csv(
    ROOT / "outputs/v9_audit/spatial_block_oof_predictions.csv"
)

assert "spatial_block" in v9.columns
assert "District" in v9.columns

folds = v9[["District", "spatial_block"]].drop_duplicates()
assert not folds.District.duplicated().any()

# Use the corrected NASA district coordinates
weather = pd.read_csv(
    ROOT / "data/03_NASA_DAILY_WEATHER_COORDINATE_CORRECTED.csv",
    usecols=["District", "Latitude", "Longitude"]
).drop_duplicates("District")

for df in (observed, folds, weather):
    df["district_key"] = df["District"].map(normalize)
    assert not df.district_key.duplicated().any()

df = observed.merge(
    folds[["district_key", "spatial_block"]],
    on="district_key",
    how="left",
    validate="one_to_one"
).merge(
    weather[["district_key", "Latitude", "Longitude"]],
    on="district_key",
    how="left",
    validate="one_to_one"
)

assert len(df) == 33
assert df[["observed", "spatial_block", "Latitude", "Longitude"]].notna().all().all()
assert df.spatial_block.nunique() == 5

predictions = []
fold_reports = []

def idw(lon, lat, values, test_lon, test_lat, power=2):
    # Haversine great-circle distances in kilometres
    r = 6371.0088
    lat1 = np.radians(np.asarray(lat))[:, None]
    lon1 = np.radians(np.asarray(lon))[:, None]
    lat2 = np.radians(np.asarray(test_lat))[None, :]
    lon2 = np.radians(np.asarray(test_lon))[None, :]
    a = (
        np.sin((lat2-lat1)/2)**2
        + np.cos(lat1)*np.cos(lat2)
        * np.sin((lon2-lon1)/2)**2
    )
    distances = 2*r*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distances = np.maximum(distances, 1e-8)
    weights = 1 / distances**power
    return (weights * np.asarray(values)[:, None]).sum(axis=0) / weights.sum(axis=0)

for block in sorted(df.spatial_block.unique()):
    train = df[df.spatial_block != block]
    test = df[df.spatial_block == block]

    assert len(train) >= 20
    assert set(train.District).isdisjoint(set(test.District))

    xtr = train.Longitude.to_numpy()
    ytr = train.Latitude.to_numpy()
    vtr = train.observed.to_numpy()

    xte = test.Longitude.to_numpy()
    yte = test.Latitude.to_numpy()

    baseline = np.full(len(test), vtr.mean())
    idw_pred = idw(xtr, ytr, vtr, xte, yte)

    try:
        kriging = OrdinaryKriging(
            xtr, ytr, vtr,
            variogram_model="spherical",
            coordinates_type="geographic",
            verbose=False,
            enable_plotting=False
        )
        kriging_pred, variance = kriging.execute("points", xte, yte)
        kriging_pred = np.asarray(kriging_pred.filled(np.nan) if hasattr(kriging_pred, "filled") else kriging_pred, dtype=float)
        variance = np.asarray(variance.filled(np.nan) if hasattr(variance, "filled") else variance, dtype=float)
        error = ""
    except Exception as exc:
        kriging_pred = np.full(len(test), np.nan)
        variance = np.full(len(test), np.nan)
        error = str(exc)

    for i, (_, row) in enumerate(test.iterrows()):
        predictions.append({
            "District": row.District,
            "spatial_block": block,
            "observed": row.observed,
            "mean_baseline": baseline[i],
            "idw_prediction": idw_pred[i],
            "kriging_prediction": kriging_pred[i],
            "kriging_variance": variance[i],
            "kriging_error": error
        })

    fold_reports.append({
        "spatial_block": block,
        "train_districts": len(train),
        "test_districts": len(test),
        "kriging_success": int(np.isfinite(kriging_pred).sum())
    })

p = pd.DataFrame(predictions)
p.to_csv(OUT / "spatial_block_kriging_predictions.csv", index=False)
pd.DataFrame(fold_reports).to_csv(
    OUT / "spatial_block_fold_checks.csv", index=False
)

metrics = []
for model, column in [
    ("Mean baseline", "mean_baseline"),
    ("IDW", "idw_prediction"),
    ("Ordinary Kriging", "kriging_prediction")
]:
    valid = np.isfinite(p[column])
    if not valid.all():
        metrics.append({
            "model": model,
            "n": int(valid.sum()),
            "RMSE": np.nan,
            "MAE": np.nan,
            "status": "FAIL_INCOMPLETE_PREDICTIONS"
        })
        continue

    metrics.append({
        "model": model,
        "n": len(p),
        "RMSE": float(np.sqrt(mean_squared_error(p.observed, p[column]))),
        "MAE": float(mean_absolute_error(p.observed, p[column])),
        "status": "COMPLETE"
    })

m = pd.DataFrame(metrics)
m.to_csv(OUT / "spatial_block_kriging_metrics.csv", index=False)

baseline_rmse = m.loc[m.model == "Mean baseline", "RMSE"].iloc[0]
kriging_rmse = m.loc[m.model == "Ordinary Kriging", "RMSE"].iloc[0]

verdict = (
    "PASS_OUTPERFORMS_MEAN_BASELINE"
    if np.isfinite(kriging_rmse) and kriging_rmse < baseline_rmse
    else "FAIL_OR_INCONCLUSIVE"
)

summary = {
    "districts": len(p),
    "spatial_blocks": int(df.spatial_block.nunique()),
    "verdict": verdict,
    "note": "Fixed five-block spatial validation. District-scale interpolation only. Not field-level hotspot validation."
}

(OUT / "QC_STATUS.json").write_text(json.dumps(summary, indent=2))
print(m.to_string(index=False))
print(json.dumps(summary, indent=2))
