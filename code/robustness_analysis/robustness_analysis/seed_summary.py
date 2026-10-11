import pickle, numpy as np, pandas as pd, collections, json
from common import *; from cvta import *
R=pickle.load(open('leiden_0_50.pkl','rb'))
THEMES={'DevSecOps':{'devsecops'},'Satellite/LEO':{'satellite','low earth orbit','satellite communication','leo satellite','satellite network'},
        'Healthcare robotics':{'robot','robotics','healthcare robot'},'Blockchain':{'blockchain'}}
rows=[];theme_rows=[];shape=[]
for ds in ['ACM','DBLP']:
    G=build_graphs(load(ds))
    for seed in range(50):
        A=R[(ds,seed)]; cvs=compute_cvs(G,A); ch,ks,links=build_chains(A); df=chain_metrics(ch,cvs); df,tau=zones(df)
        z=df.zone.value_counts(normalize=True)
        rows.append(dict(ds=ds,seed=seed,n=len(df),tau=tau,pos=np.mean(df.cmr>=0),mean_cmr=df.cmr.mean(),med_cmr=df.cmr.median(),
            maxlife=df.lifespan.max(),**{k:z.get(k,0) for k in ['IGNITION','ASCENDANT','RESIDUAL','PERIPHERAL']},
            sign_agree_slope=np.mean(np.sign(df.cmr)==np.sign(df.log_slope)),
            monotone=np.mean((df.frac_up==1)|(df.frac_up==0)), mixed=np.mean((df.frac_up>0)&(df.frac_up<1)),
            short=np.mean(df.lifespan==3)))
        for _,r in df.iterrows(): shape.append(dict(ds=ds,seed=seed,cmr=r.cmr,slope=r.log_slope,frac_up=r.frac_up,life=r.lifespan,zone=r.zone))
        for th,kws in THEMES.items():
            zs=[]
            for _,r in df.iterrows():
                y,c=r.nodes[-1]; mem={k for k,cc in A[y].items() if cc==c}
                if mem & kws: zs.append(r.zone)
            theme_rows.append(dict(ds=ds,seed=seed,theme=th,found=len(zs)>0,growing=any(z in('IGNITION','ASCENDANT') for z in zs),ign=('IGNITION' in zs)))
S=pd.DataFrame(rows); T=pd.DataFrame(theme_rows); SH=pd.DataFrame(shape)
S.to_csv('seed_summary.csv',index=False); T.to_csv('theme_summary.csv',index=False); SH.to_csv('shape.csv',index=False)
print(S.groupby('ds').agg(['mean','std','min','max']).T.round(3).to_string())
print(T.groupby(['ds','theme'])[['found','growing','ign']].mean().round(2))
