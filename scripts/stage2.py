"""Stage 2: Bayesian partial pooling with year effects; no false traditional-knowledge confirmation."""
from common import *
import pymc as pm
import arviz as az
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

d=load();a=d.dropna(subset=['cultivated']).reset_index(drop=True);X=inputs(a);y=a.cultivated.astype(int).to_numpy()
p=oof(a,X,y,lambda:pipeline(LogisticRegression(max_iter=1500,C=0.3)),True)
# Baseline uses crop prevalence from TRAINING districts only.
b=np.zeros(len(a))
for tr,te in LeaveOneGroupOut().split(X,y,a.District):
 rates=a.iloc[tr].groupby('Crop').cultivated.mean();b[te]=a.iloc[te].Crop.map(rates).fillna(y[tr].mean())
print('Cultivation LODO AUC:',roc_auc_score(y,p),'crop baseline:',roc_auc_score(y,b))
pd.DataFrame({'District':a.District,'Crop':a.Crop,'Agri_Year':a.Agri_Year,'observed':y,'predicted':p,'crop_only_baseline':b}).to_csv(OUT/'stage2_cultivation_oof.csv',index=False)
# Yield model: standardized log-yield, crop-specific slopes, year effects, district random intercepts.
# All preprocessing is descriptive for Bayesian coefficient estimation, NOT an OOF predictive evaluation.
for subset in ['preferred','strict']:
 g=d.dropna(subset=[f'yield_{subset}']).copy().reset_index(drop=True)
 if len(g)<40: print('Insufficient rows',subset);continue
 g['log_yield']=np.log(g[f'yield_{subset}'].clip(lower=1e-6))
 cols=['temperature_mean_C','rainfall_total_mm','ph_mean']
 z=g[cols].copy(); z=(z-z.mean())/z.std(ddof=0).replace(0,1)
 ci,crops=pd.factorize(g.Crop);di,districts=pd.factorize(g.District);yi,years=pd.factorize(g.Agri_Year)
 with pm.Model() as m:
  alpha=pm.Normal('alpha',0,2,shape=len(crops))
  shared=pm.Normal('shared_slope',0,0.3,shape=len(cols))
  tau=pm.HalfNormal('slope_sd',0.3,shape=len(cols))
  offsets=pm.Normal('crop_slope_offsets',0,1,shape=(len(crops),len(cols)))
  beta=pm.Deterministic('crop_slopes',shared+tau*offsets)
  d_sd=pm.HalfNormal('district_sd',0.5)
  d_offset=pm.Normal('district_offset',0,1,shape=len(districts))
  year=pm.Normal('year_effect',0,0.4,shape=len(years))
  mu=alpha[ci]+(beta[ci]*z.to_numpy()).sum(axis=1)+d_sd*d_offset[di]+year[yi]
  sigma=pm.HalfNormal('sigma',0.5)
  pm.StudentT('yield',nu=5,mu=mu,sigma=sigma,observed=g.log_yield.to_numpy())
  trace=pm.sample(750,tune=1000,chains=4,cores=2,target_accept=0.95,random_seed=42,progressbar=True)
 summ=az.summary(trace,var_names=['shared_slope','crop_slopes','district_sd','sigma'])
 summ.to_csv(OUT/f'stage2_bayesian_{subset}.csv')
 az.to_netcdf(trace,OUT/f'stage2_posterior_{subset}.nc')
 diagnostics={'max_rhat':float(summ.r_hat.max()),'min_ess_bulk':float(summ.ess_bulk.min()),'divergences':int(trace.sample_stats.diverging.sum()),'n':len(g)}
 (OUT/f'stage2_diagnostics_{subset}.json').write_text(__import__('json').dumps(diagnostics,indent=2));print(subset,diagnostics)
