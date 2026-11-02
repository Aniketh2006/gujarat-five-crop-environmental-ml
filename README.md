# Gujarat V6: stages 1–4 research code

This package upgrades the user's prior four scripts for the actual FINAL V6 schema. It does **not** claim high predictive accuracy or journal acceptance. It is an auditable starting implementation, not a validated publication result.

## Run on Ubuntu

```bash
cd Gujarat_Stage1_4_Code_V7
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/stage1.py
python scripts/stage2.py
python scripts/stage3.py
python scripts/stage4.py
```

Results go into `outputs/`. Bayesian sampling can take time. Run stages sequentially and inspect output/diagnostics. Scripts have not been tested with the full scientific dependency stack here; validate on your environment before reporting results.

## What your professor's stages mean

1. Stage 1: normalize and audit district x crop x agricultural year data; preserve raw government observations, no full-data z-score.
2. Stage 2: two-part analysis, grouped out-of-district cultivation prediction and Bayesian partially pooled yield regression with year and district effects. **Associations, not traditional claims confirmed or causality.** Crop stress threshold coefficients are intentionally not fitted because no universal, verified district- and variety-specific thresholds are available. Replace with independently sourced and pre-registered ranges only.
3. Stage 3: logistic/RF/XGBoost for cultivation; Ridge/RF/XGBoost for log yield, all with LODO predictions, crop-only and crop-year baselines. SHAP is descriptive, fitted after held-out evaluation.
4. Stage 4: exploratory 3-zone district clustering and Kriging LODO vs mean baseline. No claims of micro-hotspots at farm scale. The fixed 3-zone selection is a transparent exploratory assumption, not a proven natural number of zones.

## Critical limitations and gates

- The presence label comes only from `Observation_Status`: `REPORTED_CULTIVATION` or `REPORTED_NO_CULTIVATION`. Check that the latter means an explicitly reported zero, not absent records. An AUC from uncertain labels is not a suitability validation.
- The 682 preferred and 540 strict yield counts must be checked after loading. Both analyses are required.
- Grouping by district reduces direct leakage, but neighboring districts may still share NASA grid cells; spatial-blocked CV and temporal holdouts remain needed for a paper.
- No hyperparameter search is conducted; fixed conservative hyperparameters avoid selecting the best test score. For journal publication use nested group CV.
- Year effects in Stage 2 are descriptive. Yield Stage 3 crop-year baseline is trained on each fold's training districts only. Check crop-year availability and compare with district/year anomaly baselines.
- No irrigation, management or cultivar controls; observed associations may be confounded.
- The Stage 4 map is point-based district visualization, not a polygon suitability map. Proper polygons and geostatistical uncertainty must be incorporated before publication.
- Environmental measurements are gridded estimates, not independent field observations. Do not equate 1,031 rows with 1,031 independent spatial units.
- Do not describe this release as journal-ready until source transcription verification, label audit, posterior diagnostics, calibration, source/calendar sensitivity, spatial CV and uncertainty pass.
