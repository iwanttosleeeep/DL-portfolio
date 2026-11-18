#!/usr/bin/env python3
"""Step 2: integrate audited frozen outputs. No inference, downloads, or source writes.
Run from any directory: OPENBLAS_NUM_THREADS=1 python3 integrated_analysis/analyze_brands.py
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import csv, hashlib, json, os, platform, sys, textwrap
R=Path(__file__).resolve().parents[1]; OUT=R/'integrated_analysis'; DATA=OUT/'data'; FIG=OUT/'figures'
for p in [DATA,FIG,OUT/'runtime_cache']:p.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'runtime_cache'))
import numpy as np, pandas as pd, scipy, sklearn, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
DISPLAY={'bang-olufsen':'Bang & Olufsen','fortnum-mason':'Fortnum & Mason','herman-miller':'Herman Miller','jellycat':'Jellycat','jo-malone-london':'Jo Malone London','kodak':'Kodak','lego':'LEGO','oura':'Oura','sennheiser':'Sennheiser','stanley':'Stanley'}
BRANDS=list(DISPLAY); NAMES=list(DISPLAY.values())
def read(path):return pd.read_csv(R/path,keep_default_na=False)
def save(name,rows,index=False):pd.DataFrame(rows).to_csv(DATA/name,index=index,float_format='%.12g')
def entropy(counts):
 p=np.array(list(counts),float);p=p[p>0]/p.sum();return float(-np.dot(p,np.log2(p)))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Keep an immutable baseline of original project/audit files; new outputs stay inside OUT.
baseline=OUT/'preserved_input_hashes.csv'
if not baseline.exists():
 ps=[p for folder in ['images','results','transcripts','data','src','assignment-4','final_audit'] for p in (R/folder).rglob('*') if p.is_file() and p.name!='.DS_Store']
 ps += [R/'README.md',R/'2026_6207_lecture_L9_v2.pdf',R/'STAT6207_Instagram_Image_URL_Tracker_Cleaned.xlsx']
 with baseline.open('w') as handle:
  w=csv.writer(handle);w.writerow(['path','bytes','sha256']);w.writerows((str(p.relative_to(R)),p.stat().st_size,digest(p)) for p in sorted(ps))
inputs=['results/yolo/data/image_level_detections.csv','results/yolo/data/object_detections.csv','results/yolo/model_metadata.json','results/resnet/data/image_predictions.csv','results/resnet/data/image_embeddings.npz','results/ocr/data/ocr_image_results.csv','results/ocr/data/ocr_raw.jsonl','results/ocr/data/text_frequency.csv','assignment-4/data/manifest/dataset.csv','final_audit/FINAL_AUDIT.md','final_audit/verification_summary.json']
save('source_file_hashes.csv',[{'path':p,'sha256':digest(R/p)} for p in inputs])
y=read('results/yolo/data/image_level_detections.csv');d=read('results/yolo/data/object_detections.csv');r=read('results/resnet/data/image_predictions.csv');o=read('results/ocr/data/ocr_image_results.csv');m=read('assignment-4/data/manifest/dataset.csv');v=read('results/ocr/data/text_frequency.csv');a=np.load(R/'results/resnet/data/image_embeddings.npz',allow_pickle=False)
m['image_id']=m.brand_slug+'/'+m.image_filename
ids=list(a['image_ids']);assert len(set(ids))==400 and all(set(x.image_id)==set(ids) for x in [y,r,o,m]);assert np.array_equal(r.image_id,a['image_ids'])
for frame in [y,r,o]:assert frame.image_id.is_unique and frame.groupby('brand').size().to_dict()==dict.fromkeys(BRANDS,40)
# Full 80-class counts and image presence. Absence of a row in detections means zero, not missing.
labels=list(json.loads((R/'results/yolo/model_metadata.json').read_text())['class_labels'].values())
counts=pd.crosstab(d.brand,d.class_name).reindex(index=BRANDS,columns=labels,fill_value=0)
presence=pd.crosstab(d.image_id,d.class_name).reindex(index=ids,columns=labels,fill_value=0).gt(0).astype(int)
save('yolo_image_class_presence.csv',presence.reset_index())
composition=counts.div(counts.sum(axis=1),axis=0)
yolo_raw=1-squareform(pdist(counts.to_numpy(),metric='braycurtis'))
yolo_norm=1-squareform(pdist(composition.to_numpy(),metric='braycurtis'))
save('yolo_normalized_composition.csv',[dict(brand=b,brand_display=DISPLAY[b],class_name=c,detection_count=int(counts.loc[b,c]),detection_share=float(composition.loc[b,c]),images_with_class=int(presence.loc[[i for i in ids if i.startswith(b+'/')],c].sum()),image_prevalence=float(presence.loc[[i for i in ids if i.startswith(b+'/')],c].mean())) for b in BRANDS for c in labels])
for name,mat in [('yolo_normalized_composition_similarity.csv',yolo_norm),('yolo_raw_count_similarity_reference.csv',yolo_raw)]:save(name,pd.DataFrame(mat,index=pd.Index(BRANDS,name='brand'),columns=BRANDS),True)
# ResNet: use unique unordered within-brand pairs throughout.
f=a['features'].astype(np.float64);f/=np.linalg.norm(f,axis=1,keepdims=True);cos=f@f.T;ab=a['brands'];probs=softmax(a['logits'].astype(float),axis=1);assert np.array_equal(np.argmax(probs,axis=1),r.top1_class_id)
rsim=np.zeros((10,10));within=[];image_feature=[]
for bi,b in enumerate(BRANDS):
 idx=np.where(ab==b)[0];other=np.where(ab!=b)[0];block=cos[np.ix_(idx,idx)];vals=block[np.triu_indices(40,1)]
 within.append(dict(brand=b,brand_display=DISPLAY[b],n_images=40,unique_unordered_pairs=len(vals),within_mean_cosine=vals.mean(),within_sd_cosine=vals.std(ddof=1),within_median_cosine=np.median(vals),between_mean_cosine=cos[np.ix_(idx,other)].mean(),within_minus_between=vals.mean()-cos[np.ix_(idx,other)].mean()))
 for bj,c in enumerate(BRANDS):rsim[bi,bj]=vals.mean() if b==c else cos[np.ix_(idx,np.where(ab==c)[0])].mean()
 for k in idx:image_feature.append(dict(image_id=ids[k],resnet_mean_cosine_to_same_brand=cos[k,idx[idx!=k]].mean(),resnet_mean_cosine_to_other_brands=cos[k,other].mean()))
save('resnet_unique_pair_statistics.csv',within)
save('resnet_mean_pairwise_similarity.csv',pd.DataFrame(rsim,index=pd.Index(BRANDS,name='brand'),columns=BRANDS),True)
pairs=[]
for i,j in combinations(range(10),2):pairs.append(dict(brand_a=BRANDS[i],brand_b=BRANDS[j],brand_a_display=NAMES[i],brand_b_display=NAMES[j],yolo_normalized_composition_similarity=yolo_norm[i,j],yolo_raw_count_similarity=yolo_raw[i,j],resnet_mean_pairwise_cosine=rsim[i,j],resnet_n_image_pairs=1600))
pairs=pd.DataFrame(pairs)
for metric in ['yolo_normalized_composition_similarity','resnet_mean_pairwise_cosine']:pairs[metric+'_rank']=pairs[metric].rank(ascending=False,method='average')
save('cross_model_similarity_pairs.csv',pairs)
# A single image join supports joint evidence and source-stratification checks.
master=y[['image_id','brand','filename','object_count','zero_detection','detection_summary']].rename(columns={'object_count':'yolo_object_count','zero_detection':'yolo_zero_detection','detection_summary':'yolo_detection_summary'})
master=master.merge(r[['image_id','top1_class','top1_prob','top5_classes','top5_probs']].rename(columns={c:'resnet_'+c for c in ['top1_class','top1_prob','top5_classes','top5_probs']}),on='image_id',validate='one_to_one')
master=master.merge(o[['image_id','raw_region_count','region_count','token_count','has_text','brand_name_present','brand_mentions','text_content']].rename(columns={c:'ocr_'+c for c in ['raw_region_count','region_count','token_count','has_text','brand_name_present','brand_mentions','text_content']}),on='image_id',validate='one_to_one')
master=master.merge(m[['image_id','post_url','account_handle','account_type','account_eligibility_evidence','post_datetime_utc']],on='image_id',validate='one_to_one').merge(pd.DataFrame(image_feature),on='image_id',validate='one_to_one')
master['brand_display']=master.brand.map(DISPLAY);master['yolo_person_present']=master.image_id.map(presence['person']);master['yolo_bottle_present']=master.image_id.map(presence['bottle']);master['resnet_perfume_top1']=(master.resnet_top1_class=='perfume').astype(int)
save('image_level_integrated.csv',master)
# Brand-level profile. The repaired supplementary metric is provided, but not used as an identity score.
profile=[];top1_counts=[];w=pd.DataFrame(within).set_index('brand');globalct=r.top1_class.value_counts()
for b in BRANDS:
 g=master[master.brand==b];yr=y[y.brand==b];rr=r[r.brand==b];ct=rr.top1_class.value_counts();mode=str(ct.index[0]);nmode=int(ct.iloc[0]);ties=ct[ct==nmode].index.tolist();h=entropy(ct.values)
 dg=d[d.brand==b];dc=dg.groupby('class_name').agg(n=('image_id','size'),n_images=('image_id','nunique')).reset_index().sort_values(['n','n_images','class_name'],ascending=[False,False,True]);lead=dc.iloc[0]
 row=dict(brand=b,brand_display=DISPLAY[b],n_images=len(g),n_posts=g.post_url.nunique(),yolo_total_detections=len(dg),yolo_mean_objects=g.yolo_object_count.mean(),yolo_zero_images=int(g.yolo_zero_detection.sum()),yolo_person_images=int(g.yolo_person_present.sum()),yolo_person_pct=100*g.yolo_person_present.mean(),yolo_leading_class=lead.class_name,yolo_leading_class_images=int(lead.n_images),yolo_leading_class_image_pct=100*lead.n_images/40,resnet_modal_class=mode,resnet_modal_tied_classes=';'.join(ties),resnet_modal_images=nmode,resnet_modal_pct=100*nmode/40,resnet_unique_top1_classes=len(ct),resnet_top1_entropy_bits=h,resnet_effective_classes=2**h,resnet_mean_top1_probability=rr.top1_prob.mean(),resnet_modal_share_of_all_class_occurrences=nmode/globalct[mode],resnet_within_mean_cosine=w.loc[b,'within_mean_cosine'],resnet_within_sd_cosine_unique_pairs=w.loc[b,'within_sd_cosine'],resnet_unique_within_pairs=780,resnet_mean_cosine_to_other_brands=w.loc[b,'between_mean_cosine'],resnet_within_minus_between=w.loc[b,'within_minus_between'],ocr_text_images=int(g.ocr_has_text.sum()),ocr_text_pct=100*g.ocr_has_text.mean(),ocr_name_images=int(g.ocr_brand_name_present.sum()),ocr_name_pct=100*g.ocr_brand_name_present.mean(),ocr_total_mentions=int(g.ocr_brand_mentions.sum()),ocr_mean_regions=g.ocr_region_count.mean(),ocr_mean_tokens=g.ocr_token_count.mean(),ocr_median_tokens=g.ocr_token_count.median(),ocr_name_pct_among_retained_text_images=100*g.ocr_brand_name_present.sum()/g.ocr_has_text.sum(),missing_account_handles=int((g.account_handle=='').sum()))
 profile.append(row)
 for cls,n in ct.items():top1_counts.append(dict(brand=b,class_name=cls,n_images=int(n),pct_images=100*n/40))
profile=pd.DataFrame(profile);save('unified_brand_profile.csv',profile);save('resnet_top1_counts.csv',top1_counts)
save('corrected_resnet_modal_class_share.csv',profile[['brand','resnet_modal_class','resnet_modal_images','resnet_modal_share_of_all_class_occurrences']])
# Same-image intersections make cross-model complementarity explicit. No labels are treated as ground truth.
criteria=[('jellycat','teddy bear','teddy'),('herman-miller','chair','desk'),('jo-malone-london','bottle','perfume'),('kodak','cell phone','Polaroid camera'),('lego','person','toyshop')]
joint=[]
for b,yclass,rclass in criteria:
 g=master[master.brand==b];ay=set(d[(d.brand==b)&(d.class_name==yclass)].image_id);ar=set(g.loc[g.resnet_top1_class==rclass,'image_id']);an=set(g.loc[g.ocr_brand_name_present==1,'image_id']);union=ay|ar
 joint.append(dict(brand=b,brand_display=DISPLAY[b],yolo_class=yclass,resnet_class=rclass,n_images=40,yolo_images=len(ay),resnet_images=len(ar),both_images=len(ay&ar),yolo_only_images=len(ay-ar),resnet_only_images=len(ar-ay),either_images=len(union),neither_images=40-len(union),ocr_name_images=len(an),yolo_and_ocr_name=len(ay&an),resnet_and_ocr_name=len(ar&an),all_three=len(ay&ar&an),either_and_ocr_name=len(union&an),yolo_without_ocr_name=len(ay-an)))
save('cross_model_image_overlap.csv',joint)
# Secondary sensitivity: equal weight to each post; all images retained, no dataset replacement.
postrows=[]
for b,g in master.groupby('brand',sort=False):
 pp=g.groupby('post_url')[['yolo_person_present','ocr_has_text','ocr_brand_name_present','ocr_token_count']].mean().mean()
 for metric in pp.index:postrows.append(dict(brand=b,metric=metric,n_images=len(g),n_posts=g.post_url.nunique(),image_weighted_mean=g[metric].mean(),equal_post_weighted_mean=pp[metric],difference=pp[metric]-g[metric].mean()))
save('post_weighting_sensitivity.csv',postrows)
strata=[];jg=master[master.brand=='jo-malone-london']
for key,g in jg.groupby(jg.account_handle=='jomalonelondon',sort=False):
 strata.append(dict(source_group='official_jomalonelondon' if key else 'other_known_accounts',n_images=len(g),yolo_person_images=int(g.yolo_person_present.sum()),yolo_person_pct=100*g.yolo_person_present.mean(),yolo_bottle_images=int(g.yolo_bottle_present.sum()),resnet_perfume_images=int(g.resnet_perfume_top1.sum()),ocr_text_images=int(g.ocr_has_text.sum()),ocr_name_images=int(g.ocr_brand_name_present.sum()),ocr_name_pct=100*g.ocr_brand_name_present.mean(),ocr_mean_tokens=g.ocr_token_count.mean()))
save('jo_malone_source_strata.csv',strata)
# Top vocabulary is taken from audited term tables, preserving actual document frequencies.
vt=v[((v.ngram_type=='unigram')&(v['rank']<=5))|((v.ngram_type=='bigram')&(v['rank']<=2))].copy();save('vocabulary_evidence.csv',vt)
# Examples selected for contrasts/failure mechanisms, not as a random accuracy sample.
examples=[('jellycat/jellycat_39.jpg','A','Plush detected; visible packaging text missed','Visible Jellycat Diner packaging and plush; zero retained OCR regions.'),('herman-miller/herman-miller_06.jpg','B','Furniture agreement without name recovery','Chairs and table setting; ResNet folding chair agrees with visible chairs; OCR recognizes Ceres, not the brand.'),('kodak/kodak_26.jpg','C','Text recovers a name when visual labels fail','Seven compact cameras; YOLO cell phone and ResNet switch are category confusions; repeated Kodak text is recovered.'),('jo-malone-london/jo-malone-london_26.jpg','D','Global perfume label despite no YOLO boxes','Visible perfume presentation; YOLO has no detections, ResNet perfume and OCR target-name matching add evidence.'),('jo-malone-london/jo-malone-london_31.jpg','E','Local bottle evidence despite wrong global label','Visible fragrance/candle arrangement; YOLO bottle is useful, while ResNet Crock Pot is a global classification error.'),('sennheiser/sennheiser_35.jpg','F','Specialized product confused; no OCR rescue','Visible earbuds in a charging case; YOLO phone/remote/person and ResNet joystick do not identify the focal product.'),('fortnum-mason/fortnum-mason_22.jpg','G','Many text regions; no target-name match','Visible hamper and multiple brand labels; ResNet hamper succeeds but interleaved OCR words yield zero exact alias matches.'),('lego/lego_23.jpg','H','Toy figures trigger person detections','Visible model building and minifigures; YOLO counts 13 person detections; these are not evidence of 13 human lifestyle subjects.')]
ex=[]
for iid,panel,title,note in examples:
 row=master.set_index('image_id').loc[iid].to_dict();row.update(image_id=iid,panel=panel,example_title=title,visual_observation=note,sha256=digest(R/'images'/iid));ex.append(row)
save('representative_examples.csv',ex)
# Focused academic figures; all plotted values are exported and independently verified.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'figure.facecolor':'white'})
colours=plt.get_cmap('tab10').colors
figuremeta=[]
def finish(fig,name,source,purpose):
 fig.savefig(FIG/name,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
 with Image.open(FIG/name) as im:figuremeta.append(dict(figure=name,width=im.width,height=im.height,source_table=source,analytical_purpose=purpose))
# 1. Common denominator, different measurement tasks. Never average into a brand score.
ratecols=['yolo_person_pct','resnet_modal_pct','ocr_text_pct','ocr_name_pct'];values=profile[ratecols].to_numpy();save('figure01_comparison_data.csv',profile[['brand','brand_display']+ratecols+['ocr_mean_tokens','ocr_median_tokens']])
fig,(ax,ax2)=plt.subplots(1,2,figsize=(14.3,6.8),gridspec_kw={'width_ratios':[1.4,1]});hm=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
ax.set_yticks(range(10),NAMES);ax.set_xticks(range(4),['YOLO\nperson present','ResNet18\nmodal class','OCR\nany retained text','OCR\ntarget-name match'])
for i in range(10):
 for j in range(4):ax.text(j,i,f'{values[i,j]:g}%',ha='center',va='center',color='white' if values[i,j]>55 else '#172b4d',fontsize=10)
ax.set_title('A. Image-based measures (% of 40 images)',pad=14);fig.colorbar(hm,ax=ax,fraction=.035,pad=.025,label='% of brand images')
pos=np.arange(10);ax2.barh(pos-.17,profile.ocr_mean_tokens,height=.32,color='#167d9a',label='Mean');ax2.barh(pos+.17,profile.ocr_median_tokens,height=.32,color='#c4cbd4',label='Median');ax2.set_yticks(pos,NAMES);ax2.invert_yaxis();ax2.set_xlim(0,14)
for i,(mean,median) in enumerate(zip(profile.ocr_mean_tokens,profile.ocr_median_tokens)):ax2.text(max(mean,median)+.15,i,f'{mean:.2f} / {median:g}',va='center',fontsize=8.5)
ax2.set_xlabel('Retained OCR tokens per image');ax2.set_title('B. Text volume (mean / median)',pad=14);ax2.legend(loc='lower right');ax2.grid(axis='x',alpha=.18)
fig.suptitle('Different dimensions of the same brand image sets',fontsize=16,fontweight='bold',y=1.01);fig.text(.5,-.025,'Percentages share a denominator, not a meaning. Person labels include toy/illustrated figures; modal classes differ by brand. No combined score.',ha='center',fontsize=9);fig.tight_layout(w_pad=3)
finish(fig,'fig01_unified_brand_comparison.png','figure01_comparison_data.csv','Compare detected person context, label concentration, retained text, and name matching without equating their meaning.')
# 2. Semantic label diversity and feature cohesion need not move together.
save('figure02_diversity_cohesion_data.csv',profile[['brand','brand_display','resnet_top1_entropy_bits','resnet_unique_top1_classes','resnet_within_mean_cosine','resnet_mean_cosine_to_other_brands','resnet_within_minus_between']])
fig,(ax,ax2)=plt.subplots(1,2,figsize=(14,6.1),gridspec_kw={'width_ratios':[1.1,1]})
offsets={'bang-olufsen':(-10,10),'fortnum-mason':(-8,-18),'herman-miller':(8,8),'jellycat':(8,-4),'jo-malone-london':(-18,13),'kodak':(-8,-18),'lego':(-8,-18),'oura':(-8,-17),'sennheiser':(-18,12),'stanley':(8,-5)}
for i,row in profile.iterrows():
 ax.scatter(row.resnet_top1_entropy_bits,row.resnet_within_mean_cosine,s=78,color=colours[i],edgecolor='white',linewidth=.7,zorder=3);dx,dy=offsets[row.brand];ax.annotate(row.brand_display,(row.resnet_top1_entropy_bits,row.resnet_within_mean_cosine),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=9)
ax.set_xlim(2.4,5.35);ax.set_ylim(.598,.677);ax.grid(alpha=.18);ax.set_xlabel('Shannon entropy of top-1 class distribution (bits)');ax.set_ylabel('Mean within-brand feature cosine (780 unique pairs)');ax.set_title('A. Label diversity versus visual feature cohesion',pad=12)
ax2.barh(pos,profile.resnet_within_minus_between,color=[colours[i] for i in range(10)]);ax2.set_yticks(pos,NAMES);ax2.invert_yaxis();ax2.set_xlim(0,.093);ax2.set_xlabel('Within-brand mean minus other-brand mean cosine');ax2.grid(axis='x',alpha=.18);ax2.set_title('B. Cohesion relative to the other nine brands',pad=12)
for i,x in enumerate(profile.resnet_within_minus_between):ax2.text(x+.0012,i,f'{x:.3f}',va='center',fontsize=9)
fig.suptitle('Feature cohesion and label concentration describe different patterns',fontsize=15,fontweight='bold',y=1.02);fig.tight_layout(w_pad=2.5)
finish(fig,'fig02_diversity_and_cohesion.png','figure02_diversity_cohesion_data.csv','Show the distinction between class-distribution diversity and embedding cohesion; avoid treating entropy as total visual diversity.')
# 3. Distinct measurement definitions and scales, with common row order.
fig,axes=plt.subplots(1,2,figsize=(16.2,7));masked=np.eye(10,dtype=bool)
for ax,mat,cmap,lo,hi,title,label in [(axes[0],yolo_norm,'YlGnBu',0,1,'A. YOLO normalized object-class composition','1 - Bray-Curtis on detection shares'),(axes[1],rsim,'Purples',.55,.63,'B. ResNet18 mean cross-brand feature cosine','Mean over 1,600 cross-brand image pairs')]:
 cm=plt.get_cmap(cmap).copy();cm.set_bad('#e7eaef');im=ax.imshow(np.ma.array(mat,mask=masked),cmap=cm,vmin=lo,vmax=hi)
 ax.set_xticks(range(10),NAMES,rotation=50,ha='right',fontsize=8);ax.set_yticks(range(10),NAMES,fontsize=9);ax.set_title(title,pad=14,fontsize=12)
 for i in range(10):
  for j in range(10):
   if i!=j:ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7.7,color='white' if (mat[i,j]-lo)/(hi-lo)>.58 else '#172b4d')
 fig.colorbar(im,ax=ax,fraction=.04,pad=.02,label=label)
fig.suptitle('Brand resemblance depends on what is measured',fontsize=16,fontweight='bold',y=1.02);fig.text(.5,-.035,'Diagonal omitted. Different colour scales are intentional; the two similarities are not numerically interchangeable. Fixed alphabetical brand order.',ha='center',fontsize=9);fig.tight_layout(w_pad=3)
finish(fig,'fig03_complementary_similarity.png','yolo_normalized_composition_similarity.csv; resnet_mean_pairwise_similarity.csv','Separate object-composition resemblance from visual feature resemblance; compare rankings, not raw magnitudes.')
# 4. Diagnostic photo panel, preserving source pixels via display only.
fig=plt.figure(figsize=(17,11.8));outer=fig.add_gridspec(2,4,hspace=.19,wspace=.12)
for idx,row in enumerate(ex):
 sub=outer[idx//4,idx%4].subgridspec(2,1,height_ratios=[3.05,1.42],hspace=.04);ax=fig.add_subplot(sub[0]);tx=fig.add_subplot(sub[1]);tx.axis('off');ax.axis('off')
 ax.imshow(ImageOps.exif_transpose(Image.open(R/'images'/row['image_id'])));ax.set_title(f"{row['panel']}. {row['brand_display']} · {Path(row['image_id']).stem.rsplit('_',1)[-1]}",fontsize=11,pad=6)
 focal={'A':['teddy bear'],'B':['chair','dining table'],'C':['cell phone'],'D':[],'E':['bottle','cup'],'F':['person','cell phone','remote'],'G':['vase'],'H':['person','clock']}[row['panel']]
 current=d[d.image_id==row['image_id']].class_name.value_counts()
 detstr='; '.join(f'{c} ({int(current.get(c,0))})' for c in focal) if focal else 'no detections'
 if row['panel'] in ['B','E']:detstr+=f"; {row['yolo_object_count']} total"
 caption='\n'.join(textwrap.wrap(row['example_title'],39))+'\n'+'\n'.join(textwrap.wrap('YOLO: '+detstr,42))+f"\nResNet: {row['resnet_top1_class']} (p={row['resnet_top1_prob']:.4f})\nOCR regions: {row['ocr_region_count']}; name matches: {row['ocr_brand_mentions']}"
 tx.text(.01,.98,caption,transform=tx.transAxes,va='top',ha='left',fontsize=9,linespacing=1.3,wrap=True)
fig.suptitle('Image-level complementarity and failure mechanisms',fontsize=17,fontweight='bold',y=.975);fig.text(.5,.018,'Diagnostic examples selected after inspecting outputs. Model probabilities are not accuracy; YOLO labels and OCR strings are not verified ground truth.',ha='center',fontsize=10)
finish(fig,'fig04_representative_cross_model_examples.png','representative_examples.csv','Illustrate same-image successes, complementarity, category confusions, toy-person detections, and OCR name misses.')
save('figure_manifest.csv',figuremeta)
# Compact tables for report assembly; full precision stays in CSVs.
lines=['| Brand | Leading YOLO class: images | ResNet top-1 mode: images | H (bits) | Within cosine / gap | OCR text / name | Mean tokens |','|---|---|---|---:|---:|---:|---:|']
for q in profile.itertuples():lines.append(f'| {q.brand_display} | {q.yolo_leading_class}: {q.yolo_leading_class_images}/40 | {q.resnet_modal_class}: {q.resnet_modal_images}/40 | {q.resnet_top1_entropy_bits:.2f} | {q.resnet_within_mean_cosine:.3f} / +{q.resnet_within_minus_between:.3f} | {q.ocr_text_images}/40 / {q.ocr_name_images}/40 | {q.ocr_mean_tokens:.2f} |')
(OUT/'UNIFIED_PROFILE_TABLE.md').write_text('\n'.join(lines)+'\n')
iu=np.triu_indices(400,1);withinmask=ab[iu[0]]==ab[iu[1]]
summary=dict(n_brands=10,n_images=400,n_posts=m.post_url.nunique(),yolo_normalized_offdiagonal_mean=float(yolo_norm[np.triu_indices(10,1)].mean()),yolo_raw_count_offdiagonal_mean=float(yolo_raw[np.triu_indices(10,1)].mean()),resnet_within_unique_pairs=int(withinmask.sum()),resnet_between_unique_pairs=int((~withinmask).sum()),resnet_within_mean=float(cos[iu][withinmask].mean()),resnet_between_mean=float(cos[iu][~withinmask].mean()),similarity_pair_rank_spearman=float(spearmanr(pairs.yolo_normalized_composition_similarity,pairs.resnet_mean_pairwise_cosine).statistic),similarity_rank_correlation_note='Descriptive only; 45 brand pairs share endpoints and are not independent observations.',jo_malone_official_images=27,jo_malone_official_image_share=27/40,figures=len(figuremeta),supplementary_dominant_class_share='recomputed by explicit modal-class lookup; not used as a brand recognition score',pair_convention='780 unordered within-brand pairs; 1600 cross-brand pairs per distinct brand pair',manual_curation_note='Final manual curation was not a separate model experiment and has no separate supplied AI-session transcript.')
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'runtime_metadata.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,inference_run=False,training_run=False,randomness_used=False),indent=2)+'\n')
print(json.dumps(summary,indent=2));print(profile[['brand','resnet_modal_class','resnet_top1_entropy_bits','resnet_within_mean_cosine','ocr_text_images','ocr_name_images']].to_string(index=False))
