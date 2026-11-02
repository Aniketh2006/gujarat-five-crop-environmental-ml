from pathlib import Path
import fitz,pandas as pd,hashlib,json
root=Path(__file__).parent
pdf=Path('/mnt/data/SpicesStatisticsataGlance2021(1).pdf')
source=Path('/mnt/data/FINAL_V6_RESEARCH_RELEASE/02_ALL_GOVERNMENT_YIELD_SOURCE_EVIDENCE.csv')
p=fitz.open(pdf)
refs={'Cumin':(137,116,'8.14'),'Fennel':(139,118,'8.16'),'Fenugreek':(140,119,'8.17')}
for crop,(page,printed,table) in refs.items():
 (root/f'{crop}_Table_{table}_PDF_page_{page}.txt').write_text(p[page-1].get_text(),encoding='utf-8')
# Individually checked against readable PDF table text; no automated inference of blank cells.
checks=[('Cumin','Ahmedabad','2015-16',17600,14432,820),('Fennel','Banaskantha','2015-16',4900,10535,2150),('Fenugreek','Amreli','2015-16',106,176,1660)]
df=pd.read_csv(source)
rows=[]
for crop,district,year,area,prod,yield_p in checks:
 matches=df[(df.Crop==crop)&(df.District==district)&(df.Agri_Year==year)&(df.Source_File.astype(str).str.contains('SpicesStatistics',case=False,na=False))]
 status='MATCH' if len(matches)==1 and abs(float(matches.iloc[0].Area_ha)-area)<1e-6 and abs(float(matches.iloc[0].Production_tonnes)-prod)<1e-6 else 'REVIEW'
 rows.append(dict(crop=crop,district=district,agri_year=year,source_area_ha=area,source_production_tonnes=prod,source_published_yield_kg_ha=yield_p,source_pdf_page=refs[crop][0],printed_page=refs[crop][1],table=refs[crop][2],matching_v6_source_rows=len(matches),v6_area_ha=matches.iloc[0].Area_ha if len(matches) else None,v6_production_tonnes=matches.iloc[0].Production_tonnes if len(matches) else None,verification=status))
pd.DataFrame(rows).to_csv(root/'01_INDEPENDENT_SPOT_CHECKS.csv',index=False)
# Queue every DASD evidence record for systematic visual confirmation; avoid claiming verified by automated source linkage.
dasd=df[df.Source_File.astype(str).str.contains('SpicesStatistics',case=False,na=False)&df.Crop.isin(refs)].copy()
dasd['pdf_page']=dasd.Crop.map({k:v[0] for k,v in refs.items()})
dasd['manual_verified']='PENDING'
dasd['manual_area_from_pdf']=''
dasd['manual_production_from_pdf']=''
dasd['manual_yield_from_pdf']=''
dasd['manual_comment']=''
dasd.to_csv(root/'02_DASD_ALL_RECORDS_REVIEW_QUEUE.csv',index=False)
# Critical zero labels: these must be verified against all source years, including government Excel not present in DASD PDF.
panel=pd.read_csv('/mnt/data/FINAL_V6_RESEARCH_RELEASE/04_DISTRICT_CROP_YEAR_RESEARCH_PANEL.csv')
zero=panel[pd.to_numeric(panel.Area_ha,errors='coerce').eq(0)].copy()
zero['zero_meaning_verified']='NO'
zero['independent_source_page_check']='PENDING'
zero.to_csv(root/'03_ZERO_AREA_SOURCE_VERIFICATION_QUEUE.csv',index=False)
manifest={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'source_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pdf_pages':len(p),'review_queue_dasd_rows':len(dasd),'zero_area_queue_rows':len(zero),'independently_checked_spot_rows':len(rows),'spot_matches':sum(r['verification']=='MATCH' for r in rows),'publication_certified':False}
(root/'04_AUDIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
(root/'README.md').write_text('DASD 2021 independent audit STARTER. Three specifically checked source rows match the V6 source evidence. This is NOT a comprehensive source verification. Text extracts are for navigation; visually confirm ambiguous blanks and table cells on original PDF pages 137, 139, 140. Source columns are not auto-parsed because blank cells and merged text can shift positions. Zero area queue requires checking original government workbooks too. Do not treat missing data as zero. No V6 records have been modified.\n')
print(manifest)
