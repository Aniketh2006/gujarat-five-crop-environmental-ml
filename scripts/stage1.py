"""Stage 1: source-safe V6 preparation, no invented stress thresholds."""
import json
from common import *
d=load();check_folds(d)
assert set(d.Crop)=={'Castor','Cumin','Fennel','Fenugreek','Isabgol'}
assert d[FEATURES].notna().all().all(), 'Environmental predictor missingness must be investigated'
assert (d.loc[d.cultivated.eq(0),'yield_preferred'].isna()).all()
assert (d.loc[d.cultivated.eq(0),'yield_strict'].isna()).all()
d.to_csv(OUT/'stage1_panel.csv',index=False)
q={'rows':len(d),'districts':d.District.nunique(),'crops':d.Crop.nunique(),'years':d.Agri_Year.nunique(),'cultivation_labels':d.cultivated.value_counts(dropna=False).to_dict(),'preferred_yield_rows':int(d.yield_preferred.notna().sum()),'strict_yield_rows':int(d.yield_strict.notna().sum()),'coordinate_review_count':int(d.coordinate_review_flag.astype(str).str.contains('REVIEW|MISMATCH',case=False,na=False).sum()) if 'coordinate_review_flag' in d else None}
(OUT/'stage1_QA.json').write_text(json.dumps(q,indent=2,default=str));print(json.dumps(q,indent=2,default=str))
