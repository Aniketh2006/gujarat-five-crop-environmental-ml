from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,zipfile
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
ROOT=Path('/mnt/data'); OUT=ROOT/'research_finish'/'deliverable';OUT.mkdir(parents=True,exist_ok=True)
source=pd.read_csv(ROOT/'FINAL_V6_RESEARCH_RELEASE/02_ALL_GOVERNMENT_YIELD_SOURCE_EVIDENCE.csv')
panel=pd.read_csv(ROOT/'FINAL_V6_RESEARCH_RELEASE/04_DISTRICT_CROP_YEAR_RESEARCH_PANEL.csv')
zeros=pd.read_csv(ROOT/'DASD_Source_Audit/03_ZERO_AREA_SOURCE_VERIFICATION_QUEUE.csv')
zero_cols=['Crop','District','Agri_Year','Area_ha','Production_tonnes','Source_File','Source_Location','Source_Reference','Observation_Status','Independent_Transcription_Verification']
z=zeros[zero_cols].copy();z['arithmetic_consistency']=np.where((z.Area_ha==0)&(z.Production_tonnes==0),'ZERO_ZERO_CONSISTENT','REVIEW');z['source_cell_verification']='PENDING_INDEPENDENT_PDF_XLS_CHECK';z.to_csv(OUT/'01_ZERO_LABEL_VERIFICATION_QUEUE.csv',index=False)
summary=z.groupby(['Crop','source_cell_verification']).size().reset_index(name='n');summary.to_csv(OUT/'02_ZERO_LABEL_AUDIT_SUMMARY.csv',index=False)
# Do not silently upgrade the three prior spot checks to full independent audit.
spot=pd.read_csv(ROOT/'DASD_Source_Audit/01_INDEPENDENT_SPOT_CHECKS.csv');spot.to_csv(OUT/'03_PRIOR_SPOT_CHECKS_FOR_REVIEW.csv',index=False)
pref=pd.read_csv(ROOT/'research_finish/inputs/outputs/stage2_bayesian_preferred.csv');strict=pd.read_csv(ROOT/'research_finish/inputs/outputs/stage2_bayesian_strict.csv')
a=pref.merge(strict,on='Unnamed: 0',suffixes=('_preferred','_strict'))
a['sign_change']=np.sign(a['mean_preferred'])!=np.sign(a['mean_strict'])
a['preferred_interval_excludes_zero']=(a['hdi_3%_preferred']>0)|(a['hdi_97%_preferred']<0)
a['strict_interval_excludes_zero']=(a['hdi_3%_strict']>0)|(a['hdi_97%_strict']<0)
a['mean_change_strict_minus_preferred']=a['mean_strict']-a['mean_preferred'];a.to_csv(OUT/'04_BAYESIAN_SOURCE_SENSITIVITY.csv',index=False)
# Existing out-of-fold kriging, recalculated with same finite-pair set.
k=pd.read_csv(ROOT/'research_finish/inputs/outputs/stage4_kriging_oof.csv')
valid=k[['observed','kriging_LODO','mean_baseline_LODO']].notna().all(axis=1)
rmse_kr=float(np.sqrt(mean_squared_error(k.loc[valid,'observed'],k.loc[valid,'kriging_LODO'])))
rmse_bas=float(np.sqrt(mean_squared_error(k.loc[valid,'observed'],k.loc[valid,'mean_baseline_LODO'])))
# More demanding block holdout: train-only spatial IDW, compared to train mean. Not kriging.
district=panel[['District_norm','spatial_block']].drop_duplicates('District_norm').rename(columns={'District_norm':'District'})
weather=pd.read_csv(ROOT/'FINAL_V6_RESEARCH_RELEASE/03_NASA_DAILY_WEATHER_COORDINATE_CORRECTED.csv',usecols=['District','Latitude','Longitude']).drop_duplicates('District')
d=k[['District','observed']].merge(district,on='District',validate='one_to_one').merge(weather,on='District',validate='one_to_one')
rows=[]
for block in sorted(d.spatial_block.dropna().unique()):
 tr=d[d.spatial_block!=block];te=d[d.spatial_block==block]
 xytr=tr[['Latitude','Longitude']].to_numpy();xyte=te[['Latitude','Longitude']].to_numpy()
 dist=cdist(xyte,xytr); weights=1/np.maximum(dist,1e-6)**2
 idw=(weights@tr.observed.to_numpy())/weights.sum(axis=1)
 for (_,r),p in zip(te.iterrows(),idw):rows.append(dict(District=r.District,spatial_block=block,observed=r.observed,idw_block_pred=float(p),mean_block_pred=float(tr.observed.mean())))
b=pd.DataFrame(rows);b.to_csv(OUT/'05_SPATIAL_BLOCK_IDW_BASELINE_OOF.csv',index=False)
block_idw=float(np.sqrt(mean_squared_error(b.observed,b.idw_block_pred)))
block_base=float(np.sqrt(mean_squared_error(b.observed,b.mean_block_pred)))
metrics={'source_rows':len(source),'panel_rows':len(panel),'zero_review_rows':len(z),'zero_source_cells_independently_verified_in_this_run':0,'prior_spot_check_rows':len(spot),'bayesian_parameter_rows':len(a),'bayesian_preferred_max_rhat':float(pref.r_hat.max()),'bayesian_strict_max_rhat':float(strict.r_hat.max()),'bayesian_preferred_min_ess':float(pref.ess_bulk.min()),'bayesian_strict_min_ess':float(strict.ess_bulk.min()),'bayesian_sign_changes':int(a.sign_change.sum()),'bayesian_preferred_intervals_exclude_zero':int(a.preferred_interval_excludes_zero.sum()),'bayesian_strict_intervals_exclude_zero':int(a.strict_interval_excludes_zero.sum()),'kriging_LODO_RMSE':rmse_kr,'kriging_LODO_baseline_RMSE':rmse_bas,'spatial_block_IDW_RMSE':block_idw,'spatial_block_mean_RMSE':block_base,'spatial_block_idw_beats_mean':block_idw<block_base,'note':'Spatial-block check uses IDW, NOT kriging; blocked kriging remains untested.'}
(OUT/'06_METRICS.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
