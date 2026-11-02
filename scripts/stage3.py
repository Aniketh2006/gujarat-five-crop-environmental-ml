"""Stage 3: district-held-out ML benchmarks and descriptive SHAP (not causal)."""
from common import *
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import roc_auc_score,average_precision_score,r2_score,mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier,XGBRegressor
import shap
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

d=load(); results=[]
a=d.dropna(subset=['cultivated']).reset_index(drop=True);X=inputs(a);y=a.cultivated.astype(int).to_numpy()
for name,fn in [('logistic',lambda:pipeline(LogisticRegression(C=.3,max_iter=1500))),('random_forest',lambda:pipeline(RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=8,random_state=42,n_jobs=1))),('xgboost',lambda:pipeline(XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.04,subsample=.9,colsample_bytree=.9,reg_lambda=5,eval_metric='logloss',random_state=42,n_jobs=1)))]:
 pred=oof(a,X,y,fn,True)
 results.append({'task':'cultivation','subset':'all_labeled','model':name,'n':len(a),'AUC':roc_auc_score(y,pred),'AP':average_precision_score(y,pred)})
 pd.DataFrame({'District':a.District,'Crop':a.Crop,'Agri_Year':a.Agri_Year,'y':y,'prediction':pred}).to_csv(OUT/f'stage3_presence_{name}_oof.csv',index=False)
# Crop-only baseline, trained within each held-out split
base=np.full(len(a),np.nan)
for tr,te in LeaveOneGroupOut().split(X,y,a.District):
 rate=a.iloc[tr].groupby('Crop').cultivated.mean();base[te]=a.iloc[te].Crop.map(rate).fillna(y[tr].mean())
results.append({'task':'cultivation','subset':'all_labeled','model':'crop_only_baseline','n':len(a),'AUC':roc_auc_score(y,base),'AP':average_precision_score(y,base)})
for subset in ['preferred','strict']:
 g=d.dropna(subset=[f'yield_{subset}']).copy().reset_index(drop=True)
 # Target is log yield, avoids crop-wise global target standardization leakage.
 y=np.log(g[f'yield_{subset}'].to_numpy());Xg=inputs(g)
 baseline=np.full(len(g),np.nan)
 for tr,te in LeaveOneGroupOut().split(Xg,y,g.District):
  train=g.iloc[tr].copy();train['_y']=y[tr];rates=train.groupby(['Crop','Agri_Year'])._y.mean();cr=train.groupby('Crop')._y.mean()
  for j in te:
   r=g.iloc[j];baseline[j]=rates.get((r.Crop,r.Agri_Year),cr.get(r.Crop,y[tr].mean()))
 for name,fn in [('ridge',lambda:pipeline(Ridge(alpha=20))),('random_forest',lambda:pipeline(RandomForestRegressor(n_estimators=200,max_depth=4,min_samples_leaf=8,random_state=42,n_jobs=1))),('xgboost',lambda:pipeline(XGBRegressor(n_estimators=150,max_depth=2,learning_rate=.04,subsample=.9,colsample_bytree=.9,reg_lambda=5,random_state=42,n_jobs=1)))]:
  pred=oof(g,Xg,y,fn)
  results.append({'task':'log_yield','subset':subset,'model':name,'n':len(g),'R2':r2_score(y,pred),'MAE_log':mean_absolute_error(y,pred)})
  pd.DataFrame({'District':g.District,'Crop':g.Crop,'Agri_Year':g.Agri_Year,'log_y':y,'prediction':pred}).to_csv(OUT/f'stage3_yield_{subset}_{name}_oof.csv',index=False)
 results.append({'task':'log_yield','subset':subset,'model':'crop_year_baseline','n':len(g),'R2':r2_score(y,baseline),'MAE_log':mean_absolute_error(y,baseline)})
pd.DataFrame(results).to_csv(OUT/'stage3_model_comparison.csv',index=False)
# Fit full-data explanatory model only AFTER OOF evaluation. SHAP does not validate causality.
model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.04,reg_lambda=5,random_state=42,n_jobs=1).fit(X,y=a.cultivated.astype(int))
try:
 sv=shap.TreeExplainer(model)(X)
 shap.summary_plot(sv,X,show=False,max_display=15);plt.savefig(OUT/'stage3_SHAP_descriptive.png',dpi=150,bbox_inches='tight');plt.close()
except Exception as e: print('SHAP skipped:',repr(e))
print(pd.DataFrame(results).to_string(index=False))
