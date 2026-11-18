#!/usr/bin/env python3
"""Independent source reconciliation for Step 2 derived outputs; writes only integrated_analysis/."""
from pathlib import Path
from collections import Counter
import ast,csv,hashlib,json,re,unicodedata
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'integrated_analysis';D=O/'data';checks=[]
def rd(p):return pd.read_csv(R/p,keep_default_na=False)
def out(name):return pd.read_csv(D/name,keep_default_na=False)
def ck(name,ok,detail=''):checks.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',detail=str(detail)))
def close(name,got,expected,tol=1e-9):
 g=np.asarray(got,float);e=np.asarray(expected,float);ok=g.shape==e.shape and np.allclose(g,e,rtol=0,atol=tol,equal_nan=True)
 diff=np.abs(g-e) if g.shape==e.shape else np.array([np.inf]);ck(name,ok,f'max_abs_error={np.nanmax(diff):.5g}; atol={tol}')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
profile=out('unified_brand_profile.csv').set_index('brand');joined=out('image_level_integrated.csv').set_index('image_id');brands=list(profile.index)
y=rd('results/yolo/data/image_level_detections.csv').set_index('image_id');d=rd('results/yolo/data/object_detections.csv');r=rd('results/resnet/data/image_predictions.csv').set_index('image_id');o=rd('results/ocr/data/ocr_image_results.csv').set_index('image_id');m=rd('assignment-4/data/manifest/dataset.csv');m['image_id']=m.brand_slug+'/'+m.image_filename;m=m.set_index('image_id')
ck('400 unique joined rows / ten brands / forty each',len(joined)==400 and joined.index.is_unique and joined.groupby('brand').size().to_dict()==dict.fromkeys(brands,40))
ck('Same IDs in all input and joined tables',all(set(x.index)==set(joined.index) for x in [y,r,o,m]))
ck('350 distinct source posts',m.post_url.nunique()==350 and joined.post_url.nunique()==350)
ck('All image hashes agree with collection and three model manifests',all(sha(R/'images'/i)==m.loc[i,'sha256'] for i in joined.index))
for model,path in [('YOLO','results/yolo/audit/image_manifest_sha256.csv'),('ResNet','results/resnet/data/image_manifest_sha256.csv'),('OCR','results/ocr/data/dataset_manifest.csv')]:
 x=rd(path)
 if 'image_id' not in x:x['image_id']=x.brand+'/'+x.filename
 x=x.set_index('image_id');ck(model+' input hash agreement',set(x.index)==set(joined.index) and all(x.loc[i,'sha256']==m.loc[i,'sha256'] for i in joined.index))
for derived,source,col in [('yolo_object_count',y,'object_count'),('yolo_zero_detection',y,'zero_detection'),('resnet_top1_class',r,'top1_class'),('resnet_top1_prob',r,'top1_prob'),('resnet_top5_classes',r,'top5_classes'),('ocr_region_count',o,'region_count'),('ocr_token_count',o,'token_count'),('ocr_brand_name_present',o,'brand_name_present'),('ocr_brand_mentions',o,'brand_mentions'),('ocr_text_content',o,'text_content')]:
 if pd.api.types.is_numeric_dtype(source[col]):close('Joined '+derived,joined[derived],source.loc[joined.index,col])
 else:ck('Joined '+derived,joined[derived].astype(str).equals(source.loc[joined.index,col].astype(str)))
a=np.load(R/'results/resnet/data/image_embeddings.npz');sim=1-cdist(a['features'].astype(float),a['features'].astype(float),metric='cosine');ids=list(a['image_ids']);ab=a['brands'];rs=out('resnet_mean_pairwise_similarity.csv').set_index('brand');yp=out('yolo_normalized_composition.csv');ys=out('yolo_normalized_composition_similarity.csv').set_index('brand');rawys=out('yolo_raw_count_similarity_reference.csv').set_index('brand');wp=out('resnet_unique_pair_statistics.csv').set_index('brand');rc=out('corrected_resnet_modal_class_share.csv').set_index('brand');rglobal=Counter(r.top1_class);ysaved=rd('results/yolo/data/brand_summary.csv').set_index('brand_display');osaved=rd('results/ocr/data/brand_summary.csv').set_index('brand');rsaved=rd('results/resnet/data/brand_summary.csv').set_index('brand');presence=out('yolo_image_class_presence.csv').set_index('image_id')
for b,p in profile.iterrows():
 yb=y[y.brand==b];rb=r[r.brand==b];ob=o[o.brand==b];dg=d[d.brand==b];idx=np.where(ab==b)[0];other=np.where(ab!=b)[0];within=sim[np.ix_(idx,idx)][np.triu_indices(40,1)];cnt=Counter(rb.top1_class);prob=np.array(list(cnt.values()))/40;H=-np.dot(prob,np.log2(prob));modal=p.resnet_modal_class
 ck(b+' modal class is a maximal-frequency label',cnt[modal]==max(cnt.values()))
 for col,ex in {'n_images':40,'n_posts':m.loc[yb.index,'post_url'].nunique(),'yolo_total_detections':len(dg),'yolo_mean_objects':len(dg)/40,'yolo_zero_images':sum(yb.object_count==0),'yolo_person_images':dg.loc[dg.class_name=='person','image_id'].nunique(),'resnet_modal_images':cnt[modal],'resnet_modal_pct':100*cnt[modal]/40,'resnet_unique_top1_classes':len(cnt),'resnet_top1_entropy_bits':H,'resnet_effective_classes':2**H,'resnet_mean_top1_probability':rb.top1_prob.mean(),'resnet_modal_share_of_all_class_occurrences':cnt[modal]/rglobal[modal],'resnet_within_mean_cosine':within.mean(),'resnet_within_sd_cosine_unique_pairs':within.std(ddof=1),'resnet_unique_within_pairs':780,'resnet_mean_cosine_to_other_brands':sim[np.ix_(idx,other)].mean(),'resnet_within_minus_between':within.mean()-sim[np.ix_(idx,other)].mean(),'ocr_text_images':int(ob.has_text.sum()),'ocr_text_pct':100*ob.has_text.mean(),'ocr_name_images':int(ob.brand_name_present.sum()),'ocr_name_pct':100*ob.brand_name_present.mean(),'ocr_total_mentions':ob.brand_mentions.sum(),'ocr_mean_regions':ob.region_count.mean(),'ocr_mean_tokens':ob.token_count.mean(),'ocr_median_tokens':ob.token_count.median(),'ocr_name_pct_among_retained_text_images':100*ob.brand_name_present.sum()/ob.has_text.sum()}.items():close(b+': '+col,p[col],ex)
 close(b+' supplementary correction table',rc.loc[b,'resnet_modal_share_of_all_class_occurrences'],cnt[modal]/rglobal[modal])
 close(b+' cohesion agrees with audited unique-pair correction',wp.loc[b,'within_sd_cosine'],within.std(ddof=1))
 ck(b+' pair count convention',wp.loc[b,'unique_unordered_pairs']==780)
 # Direct detection aggregation across all 80 rows, independent of wide count-matrix construction.
 q=yp[yp.brand==b].set_index('class_name');ck(b+' eighty class rows',len(q)==80)
 close(b+' detection shares sum one',q.detection_share.sum(),1)
 for c in q.index:
  gd=dg[dg.class_name==c];close(b+'/'+c+' composition and prevalence',[q.loc[c,'detection_count'],q.loc[c,'detection_share'],q.loc[c,'images_with_class'],q.loc[c,'image_prevalence']],[len(gd),len(gd)/len(dg),gd.image_id.nunique(),gd.image_id.nunique()/40])
 lead=p.yolo_leading_class;close(b+' leading class image count',p.yolo_leading_class_images,dg[dg.class_name==lead].image_id.nunique())
 for c in brands:
  ia=np.where(ab==b)[0];ib=np.where(ab==c)[0];expected=within.mean() if b==c else sim[np.ix_(ia,ib)].mean();close(b+'/'+c+' resnet matrix',rs.loc[b,c],expected)
  arow=yp[yp.brand==b].set_index('class_name').loc[q.index];brow=yp[yp.brand==c].set_index('class_name').loc[q.index];comp=1-.5*np.abs(arow.detection_share-brow.detection_share).sum();counts=1-np.abs(arow.detection_count-brow.detection_count).sum()/(arow.detection_count.sum()+brow.detection_count.sum());close(b+'/'+c+' normalized similarity = 1-TV',ys.loc[b,c],comp);close(b+'/'+c+' raw-count reference',rawys.loc[b,c],counts)
for row in out('cross_model_image_overlap.csv').itertuples():
 bi=row.brand;images=set(y.index[y.brand==bi]);aa=set(d[(d.brand==bi)&(d.class_name==row.yolo_class)].image_id);bb=set(r.index[(r.brand==bi)&(r.top1_class==row.resnet_class)]);cc=set(o.index[(o.brand==bi)&(o.brand_name_present==1)])
 for col,ex in {'yolo_images':len(aa),'resnet_images':len(bb),'both_images':len(aa&bb),'yolo_only_images':len(aa-bb),'resnet_only_images':len(bb-aa),'either_images':len(aa|bb),'neither_images':len(images-(aa|bb)),'ocr_name_images':len(cc),'yolo_and_ocr_name':len(aa&cc),'resnet_and_ocr_name':len(bb&cc),'all_three':len(aa&bb&cc),'either_and_ocr_name':len((aa|bb)&cc),'yolo_without_ocr_name':len(aa-cc)}.items():close(bi+' overlap '+col,getattr(row,col),ex,0)
# Raw OCR kept regions/tokens, rather than trusting only derived counts.
raw=[json.loads(l) for l in (R/'results/ocr/data/ocr_raw.jsonl').open()]
for rec in raw:
 kept=[x for x in rec['detections'] if x['confidence']>=.3];nt=sum(len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?",x['text'])) for x in kept);i=rec['image_id'];close(i+' OCR from raw JSONL',[joined.loc[i,'ocr_region_count'],joined.loc[i,'ocr_token_count'],joined.loc[i,'ocr_has_text']],[len(kept),nt,int(bool(kept))],0)
# Case corrections demanded in the Step 2 request.
ck('Jellycat toyshop = 3/40',sum((r.brand=='jellycat')&(r.top1_class=='toyshop'))==3)
vc=rd('results/ocr/data/text_frequency.csv');pcs=vc[(vc.brand=='lego')&(vc.ngram_type=='unigram')&(vc.ngram=='pcs')];ck('LEGO pcs document frequency = 4/40',len(pcs)==1 and pcs.image_df.iloc[0]==4)
# Post-weighted estimates: image weights = inverse frames-per-post, then normalized by post count.
for row in out('post_weighting_sensitivity.csv').itertuples():
 g=joined[joined.brand==row.brand];sizes=g.post_url.map(g.post_url.value_counts());expected=(g[row.metric]/sizes).sum()/g.post_url.nunique();close(row.brand+' equal post weight '+row.metric,row.equal_post_weighted_mean,expected)
strata=out('jo_malone_source_strata.csv');jg=joined[joined.brand=='jo-malone-london']
for row in strata.itertuples():
 g=jg[jg.account_handle.eq('jomalonelondon') if row.source_group=='official_jomalonelondon' else ~jg.account_handle.eq('jomalonelondon')]
 for col,expected in {'n_images':len(g),'yolo_person_images':int(g.yolo_person_present.sum()),'yolo_person_pct':100*g.yolo_person_present.mean(),'yolo_bottle_images':int(g.yolo_bottle_present.sum()),'resnet_perfume_images':int(g.resnet_perfume_top1.sum()),'ocr_text_images':int(g.ocr_has_text.sum()),'ocr_name_images':int(g.ocr_brand_name_present.sum()),'ocr_name_pct':100*g.ocr_brand_name_present.mean(),'ocr_mean_tokens':g.ocr_token_count.mean()}.items():close(row.source_group+': '+col,getattr(row,col),expected)
ck('Jo official / other strata = 27 / 13',strata.set_index('source_group').n_images.to_dict()=={'official_jomalonelondon':27,'other_known_accounts':13})
# Figure-source tables match the verified master profile; render QA is recorded separately.
for name in ['figure01_comparison_data.csv','figure02_diversity_cohesion_data.csv']:
 figdata=out(name).set_index('brand')
 for col in figdata.columns:
  if col!='brand_display':close(name+': '+col,figdata[col],profile.loc[figdata.index,col])
ex=out('representative_examples.csv').set_index('image_id')
for i,row in ex.iterrows():
 ck(i+' example image hash',row.sha256==sha(R/'images'/i))
 close(i+' example displayed numbers',[row.yolo_object_count,row.resnet_top1_prob,row.ocr_region_count,row.ocr_brand_mentions],[y.loc[i,'object_count'],r.loc[i,'top1_prob'],o.loc[i,'region_count'],o.loc[i,'brand_mentions']])
 ck(i+' example label',row.resnet_top1_class==r.loc[i,'top1_class'])
manifest=out('figure_manifest.csv');ck('Four PNG figures',len(manifest)==4 and len(list((O/'figures').glob('*.png')))==4)
for row in manifest.itertuples():
 with Image.open(O/'figures'/row.figure) as im:
  im.load();ck(row.figure+' dimensions and PNG integrity',im.size==(row.width,row.height) and im.format=='PNG' and min(im.size)>=1500)
# Each vocabulary entry is exactly tied to its audited count/frequency, including corrected pcs.
vo=out('vocabulary_evidence.csv');vsource=vc.set_index(['brand','ngram_type','ngram'])
for row in vo.itertuples():
 src=vsource.loc[(row.brand,row.ngram_type,row.ngram)];close(row.brand+'/'+row.ngram+' vocabulary',[row.term_count,row.image_df,row.image_pct],[src.term_count,src.image_df,src.image_pct])
# Explicit statistical invariants and summary checks.
summary=json.loads((O/'analysis_summary.json').read_text());close('Normalized similarity mean',summary['yolo_normalized_offdiagonal_mean'],ys.to_numpy()[np.triu_indices(10,1)].mean());pairs=out('cross_model_similarity_pairs.csv');ck('45 distinct brand pairs',len(pairs)==45 and not pairs.duplicated(['brand_a','brand_b']).any())
for row in pairs.itertuples():
 close(row.brand_a+'/'+row.brand_b+' pair sources',[row.yolo_normalized_composition_similarity,row.yolo_raw_count_similarity,row.resnet_mean_pairwise_cosine],[ys.loc[row.brand_a,row.brand_b],rawys.loc[row.brand_a,row.brand_b],rs.loc[row.brand_a,row.brand_b]])
close('Similarity rank Spearman descriptive statistic',summary['similarity_pair_rank_spearman'],spearmanr(pairs.yolo_normalized_composition_similarity,pairs.resnet_mean_pairwise_cosine).statistic)
ck('Unique pair totals 7800 within / 72000 between',summary['resnet_within_unique_pairs']==7800 and summary['resnet_between_unique_pairs']==72000)
# All original and audit inputs must stay untouched; only the new output directory is writable by these scripts.
base=pd.read_csv(O/'preserved_input_hashes.csv');changed=[]
for row in base.itertuples():
 p=R/row.path
 if not p.is_file() or sha(p)!=row.sha256:changed.append(row.path)
ck('All original project and final-audit files preserved',not changed,f'baseline={len(base)}; changed/missing={changed}')
known=set(base.path);added=[str(p.relative_to(R)) for folder in ['images','results','transcripts','data','src','assignment-4','final_audit'] for p in (R/folder).rglob('*') if p.is_file() and p.name!='.DS_Store' and str(p.relative_to(R)) not in known]
ck('No files added in original input folders',not added,str(added))
# Match the original independent audit baseline too; do not rerun its writer.
old=pd.read_csv(R/'final_audit/original_file_hashes.csv');ck('Original audit baseline of 1090 unchanged',len(old)==1090 and all((R/row.path).is_file() and sha(R/row.path)==row.sha256 for row in old.itertuples()))
pd.DataFrame(checks).to_csv(O/'verification_checks.csv',index=False)
result=dict(checks=len(checks),passed=sum(c['status']=='PASS' for c in checks),failed=sum(c['status']=='FAIL' for c in checks),failed_checks=[c for c in checks if c['status']=='FAIL'],preserved_original_and_audit_files=len(base),original_audit_baseline_files=len(old),new_statistics_source='frozen prediction tables, OCR raw JSONL and ResNet18 embeddings; no inference run',figure_review_record='FIGURE_REVIEW.md',scope='Verification is numerical consistency and visual QA, not model accuracy or source authentication.')
(O/'verification_summary.json').write_text(json.dumps(result,indent=2)+'\n');(O/'VERIFICATION.md').write_text(f"""# Step 2 verification

**{result['passed']:,}/{result['checks']:,} checks passed; {result['failed']} failed.**

The verifier reads original prediction/detection tables, raw OCR JSONL, embeddings, image files, and the new outputs. It does not execute model inference or the original experiment analysis/audit scripts.

Verified coverage: all 400 image joins and hashes; 10 brands with 40 images each and 350 distinct posts; the unified profile; all 80-class normalized distributions and both YOLO similarity definitions; corrected modal-class shares; 780 unique unordered feature pairs per brand; all 45 cross-brand pairs; same-image intersections; raw retained OCR region/token counts; vocabulary evidence including LEGO pcs=4/40; source strata and post weighting; plotted data and PNG integrity. Figure layout and interpretation were reviewed separately in FIGURE_REVIEW.md.

Preservation: **{result['preserved_original_and_audit_files']:,} original project/audit files remain unchanged**, including all {result['original_audit_baseline_files']:,} files from the original audit baseline. No new files were added in protected input folders. All new outputs remain in integrated_analysis/.

Corrections applied: see corrections_applied.csv and BRAND_ANALYSIS.md. The final manual curation stage is described as manual preparation, not a missing model experiment. No final PDF was generated.

Machine-readable detail: verification_checks.csv and verification_summary.json. Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 integrated_analysis/verify_analysis.py.

Passing checks establish arithmetic/source agreement and preservation, not recognition accuracy, representative sampling, or historical source authentication. The 45 similarity pairs are dependent; the reported rank correlation is descriptive only.
""");print(json.dumps(result,indent=2));sys_exit=1 if result['failed'] else 0
raise SystemExit(sys_exit)
