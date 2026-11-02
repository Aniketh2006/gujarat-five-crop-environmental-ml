from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
PANEL=DATA/'04_DISTRICT_CROP_YEAR_RESEARCH_PANEL.csv'
FEATURES=['temperature_mean_C','temperature_max_mean_C','rainfall_total_mm','humidity_mean_pct','days_tmax_gt_35C','ph_mean','sand_mean','clay_mean','soc_0cm_g_kg']
def load():
 d=pd.read_csv(PANEL)
 d['District_source_label']=d['District']
 d['District']=d['District_norm']
 assert d.District.nunique()==33, 'Expected exactly 33 normalized districts'
 assert not d.duplicated(['District','Crop','Agri_Year']).any(), 'Duplicate district/crop/year'
 d['year_start']=pd.to_numeric(d.Agri_Year.str.slice(0,4),errors='coerce')
 for c in FEATURES+['Area_ha','Production_tonnes','Preferred_Yield_kg_ha','Strict_Model_Yield_kg_ha']:
  d[c]=pd.to_numeric(d[c],errors='coerce')
 d['cultivated']=np.where(d.Observation_Status.eq('REPORTED_CULTIVATION'),1,np.where(d.Observation_Status.eq('REPORTED_NO_CULTIVATION'),0,np.nan))
 d['yield_preferred']=d.Preferred_Yield_kg_ha.where(d.preferred_eligible.eq(1)&d.Observation_Status.eq('REPORTED_CULTIVATION'))
 d['yield_strict']=d.Strict_Model_Yield_kg_ha.where(d.strict_eligible.eq(1)&d.Observation_Status.eq('REPORTED_CULTIVATION'))
 return d

def check_folds(d):
 from sklearn.model_selection import LeaveOneGroupOut
 for tr,te in LeaveOneGroupOut().split(d,groups=d.District):
  assert not set(d.iloc[tr].District)&set(d.iloc[te].District)
 return True

def inputs(d, cols=FEATURES):
 return pd.concat([d[cols].replace([np.inf,-np.inf],np.nan),pd.get_dummies(d.Crop,prefix='crop',dtype=float)],axis=1)

def pipeline(model):
 from sklearn.pipeline import make_pipeline
 from sklearn.impute import SimpleImputer
 from sklearn.preprocessing import StandardScaler
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=False),StandardScaler(),model)

def oof(d,X,y,model_factory,classification=False):
 from sklearn.model_selection import LeaveOneGroupOut
 pred=np.full(len(d),np.nan)
 for tr,te in LeaveOneGroupOut().split(X,y,d.District):
  if classification and len(np.unique(y[tr]))<2:
   pred[te]=np.mean(y[tr]);continue
  model=model_factory();model.fit(X.iloc[tr],y[tr]);pred[te]=model.predict_proba(X.iloc[te])[:,1] if classification else model.predict(X.iloc[te])
 return pred
