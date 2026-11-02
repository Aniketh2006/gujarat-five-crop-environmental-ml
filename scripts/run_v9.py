from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs'/'stage1_panel.csv'
if not DATA.exists():
    DATA=ROOT/'data'/'04_DISTRICT_CROP_YEAR_RESEARCH_PANEL.csv'
if not DATA.exists():
    raise FileNotFoundError('Put V9 script in scripts/ and run Stage 1 first, or keep data/04_DISTRICT_CROP_YEAR_RESEARCH_PANEL.csv')
OUT=ROOT/'outputs'/'v9_audit';OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA)
if 'cultivated' not in df:
    df['cultivated']=np.where(df['Area_ha'].notna(),(df['Area_ha']>0).astype(float),np.nan)
cols=['temperature_mean_C','rainfall_total_mm','humidity_mean_pct','ph_mean','soc_0cm_g_kg','sand_mean','clay_mean']
missing=[c for c in cols+['Crop','District','spatial_block','cultivated','Area_ha'] if c not in df]
if missing:raise ValueError('Missing columns: '+str(missing))
rows=df[df.cultivated.isin([0,1])].copy().reset_index(drop=True)
rows['spatial_block']=rows['spatial_block'].astype(str)
if rows.groupby('District').spatial_block.nunique().max()!=1:raise ValueError('District assigned multiple blocks')
if rows.duplicated(['District','Crop','Agri_Year']).any():raise ValueError('Duplicate keys in V9 input')
rows['label_area_disagreement']=((rows.Area_ha>0).astype(int)!=rows.cultivated.astype(int))
rows[['District','Crop','Agri_Year','Area_ha','cultivated','label_area_disagreement','Source_File','Source_Location']].to_csv(OUT/'label_audit.csv',index=False)
num=cols

def model(use_environment):
    trans=[]
    if use_environment:trans.append(('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num))
    trans.append(('crop',OneHotEncoder(handle_unknown='ignore'),['Crop']))
    return Pipeline([('pre',ColumnTransformer(trans)),('clf',LogisticRegression(max_iter=2000,C=1.0))])

pred=np.full((len(rows),2),np.nan)
folds=[]
for block in sorted(rows.spatial_block.unique()):
    te=np.flatnonzero(rows.spatial_block.eq(block));tr=np.flatnonzero(~rows.spatial_block.eq(block))
    if not len(te) or not len(tr):continue
    if set(rows.District.iloc[te]) & set(rows.District.iloc[tr]):raise AssertionError('District leakage')
    if rows.cultivated.iloc[tr].nunique()<2:raise ValueError('One-class training fold')
    for j,env in enumerate([False,True]):
        m=model(env);m.fit(rows.iloc[tr],rows.cultivated.iloc[tr]);pred[te,j]=m.predict_proba(rows.iloc[te])[:,1]
    folds.append(dict(block=block,train_rows=len(tr),test_rows=len(te),train_districts=rows.District.iloc[tr].nunique(),test_districts=rows.District.iloc[te].nunique(),test_positive=int(rows.cultivated.iloc[te].sum()),test_negative=int(len(te)-rows.cultivated.iloc[te].sum())))
if np.isnan(pred).any():raise ValueError('Unpredicted rows')
rows['crop_only_oof']=pred[:,0];rows['environment_plus_crop_oof']=pred[:,1]
rows.to_csv(OUT/'spatial_block_oof_predictions.csv',index=False)
pd.DataFrame(folds).to_csv(OUT/'spatial_block_fold_checks.csv',index=False)
y=rows.cultivated.astype(int).to_numpy()
def score(v):return dict(AUC=float(roc_auc_score(y,v)),AP=float(average_precision_score(y,v)),Brier=float(brier_score_loss(y,v)))
metrics={'n':len(rows),'districts':int(rows.District.nunique()),'blocks':len(folds),'area_label_disagreements':int(rows.label_area_disagreement.sum()),'crop_only':score(pred[:,0]),'environment_plus_crop':score(pred[:,1])}
metrics['delta_auc']=metrics['environment_plus_crop']['AUC']-metrics['crop_only']['AUC']
# Cluster bootstrap resamples entire districts; predictions remain fixed out-of-fold predictions.
rng=np.random.default_rng(20261029);districts=rows.District.unique();ix={d:np.flatnonzero(rows.District.eq(d)) for d in districts};boots=[]
for _ in range(500):
    draw=rng.choice(districts,len(districts),replace=True);ids=np.concatenate([ix[d] for d in draw]);yy=y[ids]
    if np.unique(yy).size<2:continue
    boots.append([roc_auc_score(yy,pred[ids,0]),roc_auc_score(yy,pred[ids,1])])
b=np.array(boots)
metrics['bootstrap_95pct']={'crop_only':np.quantile(b[:,0],[.025,.975]).tolist(),'environment_plus_crop':np.quantile(b[:,1],[.025,.975]).tolist(),'delta_auc':np.quantile(b[:,1]-b[:,0],[.025,.975]).tolist(),'replicates':len(b)}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
(OUT/'README.md').write_text('V9 spatial-block audit. Out-of-fold predictions use 5 pre-existing geographic blocks. All imputation and scaling fitted on training blocks. No hyperparameter search. The blocks are provisional; AUC assesses reported cultivation, not intrinsic agronomic suitability. Zero-area source records require independent verification. Bootstrap intervals condition on fixed model predictions and blocks. AUC should not be interpreted as external validation.\n')
print(json.dumps(metrics,indent=2))
