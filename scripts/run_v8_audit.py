from pathlib import Path
import pandas as pd, numpy as np, json, zipfile, shutil
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
root=Path(__file__).resolve().parents[1]; out=root/'outputs'/'v8_audit';out.mkdir(parents=True,exist_ok=True)
outputs=root/'outputs';panel=pd.read_csv(outputs/'stage1_panel.csv',low_memory=False)
report={'rows':len(panel),'columns':list(panel.columns),'checks':{}}
for col in ['Crop','District','Agri_Year','cultivated','yield_preferred','yield_strict']:
 if col in panel: report['checks'][col]={'non_null':int(panel[col].notna().sum()),'unique':int(panel[col].nunique())}
# identify label logic vs missing area
for col in ['Area_ha','Production_tonnes','Observation_Status','cultivated']:
 if col not in panel: continue
label=panel[['Crop','District','Agri_Year','cultivated','Area_ha','Production_tonnes','Observation_Status']].copy()
label['area_missing']=label.Area_ha.isna();label['area_zero']=label.Area_ha.eq(0);label['area_positive']=label.Area_ha.gt(0)
label['label_inconsistent_area']=((label.cultivated.eq(0)&label.area_positive)|(label.cultivated.eq(1)&label.area_zero))
label['zero_with_missing_area']=label.cultivated.eq(0)&label.area_missing
label.to_csv(out/'01_CULTIVATION_LABEL_AUDIT.csv',index=False)
report['label_audit']={'inconsistent_area':int(label.label_inconsistent_area.sum()),'zero_with_missing_area':int(label.zero_with_missing_area.sum()),'zero_labels':int(label.cultivated.eq(0).sum())}
# audit existing oof predictions and crop-year residual skill
rows=[]
for subset in ['preferred','strict']:
 files=sorted(outputs.glob(f'stage3_yield_{subset}_*_oof.csv'))
 for f in files:
  d=pd.read_csv(f)
  if d.empty:continue
  rows.append({'task':'yield','subset':subset,'model':f.stem.split(subset+'_')[-1], 'n':len(d),'r2_pooled':r2_score(d.log_y,d.prediction),'mae_log':mean_absolute_error(d.log_y,d.prediction),'r2_within_crop_year': 1-((d.log_y-d.prediction)**2).sum()/max(1e-12,((d.log_y-d.groupby(['Crop','Agri_Year']).log_y.transform('mean'))**2).sum())})
 for f in files:
  d=pd.read_csv(f); d['error']=d.log_y-d.prediction
  d.groupby(['Crop','Agri_Year']).agg(n=('error','size'),mean_error=('error','mean'),mae=('error',lambda x:np.mean(np.abs(x)))).reset_index().to_csv(out/f'02_ERROR_BY_CROP_YEAR_{subset}_{f.stem.split(subset+"_")[-1]}.csv',index=False)
for f in sorted(outputs.glob('stage3_presence_*_oof.csv')):
 d=pd.read_csv(f); rows.append({'task':'cultivation','subset':'all','model':f.stem.replace('stage3_presence_',''),'n':len(d),'auc':roc_auc_score(d.y,d.prediction)})
pd.DataFrame(rows).to_csv(out/'03_RECOMPUTED_OOF_METRICS.csv',index=False)
# district cluster bootstrap: uncertainty without claiming independent repeated years
rng=np.random.default_rng(2026); boots=[]
for f in sorted(outputs.glob('stage3_presence_*_oof.csv')):
 d=pd.read_csv(f);districts=d.District.unique();vals=[]
 for _ in range(1000):
  sample=rng.choice(districts,size=len(districts),replace=True)
  z=pd.concat([d[d.District.eq(s)] for s in sample],ignore_index=True)
  if z.y.nunique()==2:vals.append(roc_auc_score(z.y,z.prediction))
 boots.append({'model':f.stem,'n_boot':len(vals),'auc':roc_auc_score(d.y,d.prediction),'ci95_lo':np.quantile(vals,.025),'ci95_hi':np.quantile(vals,.975)})
pd.DataFrame(boots).to_csv(out/'04_DISTRICT_BOOTSTRAP_AUC.csv',index=False)
# conditional yield anomaly experiment: group means TRAIN ONLY and residualize by crop-year, evaluate unseen districts
# features selected only from existing documented columns
candidates=['rainfall_total_mm','temperature_mean_c','temperature_max_mean_c','humidity_mean_pct','heat_days_gt_35','ph','sand','clay','soc_g_kg','rainy_days']
cols=[c for c in candidates if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
if len(cols)<2:
 cols=[c for c in panel.columns if c.lower() in ['rainfall_total_mm','temperature_mean','humidity_mean','ph_mean','sand_mean','clay_mean','soc_g_kg'] and pd.api.types.is_numeric_dtype(panel[c])]
experiment=[]; predouts=[]
for subset in ['preferred','strict']:
 g=panel.loc[panel[f'yield_{subset}'].gt(0)].copy().reset_index(drop=True)
 if len(g)<50 or len(cols)<2: continue
 g['log_y']=np.log(g[f'yield_{subset}'])
 groups=g.District.to_numpy(); pred0=np.full(len(g),np.nan); pred1=pred0.copy()
 for tr,te in LeaveOneGroupOut().split(g,groups=groups):
  a=g.iloc[tr].copy(); b=g.iloc[te].copy()
  mu=a.groupby(['Crop','Agri_Year']).log_y.mean();cmu=a.groupby('Crop').log_y.mean();glob=a.log_y.mean()
  a['base']=[mu.get((r.Crop,r.Agri_Year),cmu.get(r.Crop,glob)) for r in a.itertuples()]
  bbase=np.array([mu.get((r.Crop,r.Agri_Year),cmu.get(r.Crop,glob)) for r in b.itertuples()]);pred0[te]=bbase
  # training only: climate anomalies relative to crop-year TRAIN means
  trainmeans=a.groupby(['Crop','Agri_Year'])[cols].mean()
  for c in cols:
   a[c+'_anom']=[getattr(r,c)-trainmeans.loc[(r.Crop,r.Agri_Year),c] if (r.Crop,r.Agri_Year) in trainmeans.index else np.nan for r in a.itertuples()]
   b[c+'_anom']=[getattr(r,c)-trainmeans.loc[(r.Crop,r.Agri_Year),c] if (r.Crop,r.Agri_Year) in trainmeans.index else np.nan for r in b.itertuples()]
  feats=[c+'_anom' for c in cols]
  pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),('ridge',Ridge(alpha=100))])
  pipe.fit(a[feats],a.log_y-a.base)
  pred1[te]=bbase+pipe.predict(b[feats])
 for model,pred in [('crop_year_baseline',pred0),('ridge_climate_anomaly',pred1)]:
  experiment.append({'subset':subset,'model':model,'n':len(g),'r2_pooled':r2_score(g.log_y,pred),'mae_log':mean_absolute_error(g.log_y,pred),'rmse_log':np.sqrt(mean_squared_error(g.log_y,pred))})
 predouts.append(pd.DataFrame({'District':g.District,'Crop':g.Crop,'Agri_Year':g.Agri_Year,'log_y':g.log_y,'baseline_oof':pred0,'anomaly_oof':pred1,'subset':subset}))
pd.DataFrame(experiment).to_csv(out/'05_ANOMALY_EXPERIMENT_RESULTS.csv',index=False)
if predouts:pd.concat(predouts).to_csv(out/'06_ANOMALY_OOF_PREDICTIONS.csv',index=False)
report['anomaly_features']=cols
(out/'07_AUDIT_SUMMARY.json').write_text(json.dumps(report,indent=2,default=str))
