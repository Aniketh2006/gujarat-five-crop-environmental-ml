"""Stage 4: district-scale environmental zones and spatially held-out interpolation."""
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,adjusted_rand_score,mean_squared_error
from sklearn.model_selection import LeaveOneOut
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

d=load()
# District mean environmental features; never use yield to construct zones.
cols=['temperature_mean_C','rainfall_total_mm','humidity_mean_pct','ph_mean','sand_mean','clay_mean','soc_0cm_g_kg']
agg=d.groupby('District')[cols].mean().dropna(); assert len(agg)>=25
X=StandardScaler().fit_transform(agg)
# Fixed K=3 exploratory zoning; no outcome-based tuning.
labels=AgglomerativeClustering(n_clusters=3,linkage='ward').fit_predict(X)
zones=agg.copy();zones['zone']=labels;zones.to_csv(OUT/'stage4_district_zones.csv')
print('Silhouette:',silhouette_score(X,labels))
# Validate zone association against OUTCOMES using district-aggregated observed cultivation only.
obs=d.dropna(subset=['cultivated']).groupby('District').cultivated.mean()
t=zones.join(obs.rename('cultivation_rate')).dropna()
def eta(values,lab):
 overall=values.mean();den=((values-overall)**2).sum()
 return sum((values[lab==i].size)*(values[lab==i].mean()-overall)**2 for i in np.unique(lab))/den if den>0 else np.nan
rng=np.random.default_rng(42);v=t.cultivation_rate.to_numpy();lab=t.zone.to_numpy();actual=eta(v,lab)
p=(1+sum(eta(v,rng.permutation(lab))>=actual for _ in range(999)))/1000
pd.DataFrame([{'metric':'district_cultivation_rate','eta_squared':actual,'permutation_p_exploratory':p,'n_districts':len(t)}]).to_csv(OUT/'stage4_zone_validation.csv',index=False)
# Spatial interpolation is NOT micro-hotspot inference; use coordinate data from the daily NASA file.
w=pd.read_csv(DATA/'03_NASA_DAILY_WEATHER_COORDINATE_CORRECTED.csv',usecols=['District','Latitude','Longitude']).drop_duplicates('District').set_index('District')
s=t.join(w).dropna(subset=['Latitude','Longitude'])
xx=s.Longitude.to_numpy(); yy=s.Latitude.to_numpy(); vv=s.cultivation_rate.to_numpy()
base=[];kr=[]
for i in range(len(vv)):
 tr=np.arange(len(vv))!=i
 base.append(np.mean(vv[tr]))
 try:
  model=OrdinaryKriging(xx[tr],yy[tr],vv[tr],variogram_model='spherical',coordinates_type='geographic',verbose=False,enable_plotting=False)
  kr.append(float(model.execute('points',np.array([xx[i]]),np.array([yy[i]]))[0][0]))
 except Exception:kr.append(np.nan)
valid=np.isfinite(kr)
summary={'n_districts':len(vv),'n_kriging_success':int(valid.sum()),'baseline_RMSE':float(np.sqrt(mean_squared_error(vv,base))),'kriging_RMSE':float(np.sqrt(mean_squared_error(vv[valid],np.array(kr)[valid]))) if valid.sum()>=20 else np.nan,'note':'Illustrative district-scale interpolation; not field-level hotspots. No variogram tuning on test districts.'}
pd.DataFrame([summary]).to_csv(OUT/'stage4_kriging_validation.csv',index=False)
pd.DataFrame({'District':s.index,'observed':vv,'kriging_LODO':kr,'mean_baseline_LODO':base}).to_csv(OUT/'stage4_kriging_oof.csv',index=False)
fig,ax=plt.subplots(figsize=(8,6));sc=ax.scatter(xx,yy,c=vv,s=75);fig.colorbar(sc,ax=ax,label='Observed cultivation rate');ax.set(xlabel='Longitude',ylabel='Latitude',title='District-scale observed cultivation (not micro-hotspots)');fig.tight_layout();fig.savefig(OUT/'stage4_district_map.png',dpi=170)
print(summary)
