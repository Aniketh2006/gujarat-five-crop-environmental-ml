# V9 spatial validation upgrade

This patch is additive. Do not overwrite V7 scripts or datasets.

## Ubuntu installation
From `~/Music/project/Gujarat_Stage1_4_Code_V7`:

```bash
source .venv/bin/activate
mkdir -p scripts
# Copy run_v9.py from this ZIP into scripts/
python scripts/run_v9.py
zip -r V9_results.zip outputs/v9_audit
```

## Independently executed result on supplied V7 Stage 1 panel
- 1,031 rows; 33 districts; 5 fixed geographic blocks.
- Geographic-block AUC: crop-only 0.7436; crop + environment 0.9063.
- Difference +0.1627; 500 district bootstrap replicates 95% interval [0.0811, 0.2396].
- No within-fold district overlap; 0 label-versus-area arithmetic disagreements.

## Caveats
- Geographic blocks were pre-existing and provisional, not independently certified spatial buffers.
- This is reported cultivation occurrence, not agronomic suitability, and absence labels need manual government-source confirmation.
- Some district-year weather points share grid cells; climate resolution limits effective spatial information.
- The interval is a district-resampling interval conditional on these fixed out-of-fold predictions, not a guarantee of transport to other regions or years.
- No claim of yield skill or journal readiness is made.
- The data and labels must be checked against original government source tables.

## Next work
1. Verify all source zero-area records and stratified sample of positive-area rows, keeping page/cell and discrepancy log.
2. Check Stage 2 posterior predictive fit and crop-specific effects on preferred vs strict subsets.
3. Evaluate Stage 4 Kriging with geographic-blocked testing and uncertainty calibration, and ensure mapped region uses actual Gujarat polygon.
4. Prepare course report now; decide on irrigation/management additions for future journal extension.
