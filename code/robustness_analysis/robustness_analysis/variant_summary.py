import pickle, numpy as np, pandas as pd, collections, itertools
from sklearn.metrics import adjusted_mutual_info_score as AMI
from common import *; from variants import *
V=pickle.load(open('variants_V1_V2_V3.pkl','rb')); L=pickle.load(open('leiden_0_50.pkl','rb'))
THEMES={'DevSecOps':{'devsecops'},'Satellite/LEO':{'satellite','low earth orbit','satellite communication','leo satellite','satellite network'},
        'Healthcare robotics':{'robot','robotics'},'Blockchain':{'blockchain'}}
rows=[]
for ds in ['ACM','DBLP']:
    P=load(ds); G0=build_graphs(P)
    cmaps={'V1':canon_map(P,'cleaned_keywords')}
    for v in ['V0','V1','V2','V3']:
        for seed in range(10):
            if v=='V0':
                A=L[(ds,seed)]; cvs=compute_cvs(G0,A); from cvta import build_chains, chain_metrics, zones
                ch,_,_=build_chains(A); df=chain_metrics(ch,cvs); df,tau=zones(df)
            else:
                r=V[(ds,v,seed)]; A,df,tau=r['A'],r['df'],r['tau']
            th={}
            for t,kws in THEMES.items():
                zs=[]
                for _,c in df.iterrows():
                    y,cid=c.nodes[-1]; mem={k for k,cc in A[y].items() if cc==cid}
                    if mem & kws: zs.append(c.zone)
                th[t]=any(z in ('IGNITION','ASCENDANT') for z in zs)
            # partition agreement with V0 (same seed) on shared keywords, averaged over years
            a0=L[(ds,seed)]; amis=[]
            for y in A:
                if v=='V1': m={k:cmaps['V1'].get(k,k) for k in a0[y]}
                else: m={k:k for k in a0[y]}
                ks=[k for k in a0[y] if m[k] in A[y]]
                if len(ks)>10: amis.append(AMI([a0[y][k] for k in ks],[A[y][m[k]] for k in ks]))
            rows.append(dict(ds=ds,v=v,seed=seed,n=len(df),pos=np.mean(df.cmr>=0),medcmr=df.cmr.median(),
                **{z:np.mean(df.zone==z) for z in ['IGNITION','ASCENDANT','RESIDUAL','PERIPHERAL']},AMI_vs_V0=np.mean(amis),**th))
D=pd.DataFrame(rows); D.to_csv('variant_summary.csv',index=False)
pd.set_option('display.width',250)
print(D.groupby(['ds','v']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
# reference: seed-to-seed AMI within V0
ref=[]
for ds in ['ACM','DBLP']:
    for s1,s2 in [(0,1),(2,3),(4,5),(6,7),(8,9)]:
        a,b=L[(ds,s1)],L[(ds,s2)]
        ref.append(dict(ds=ds,ami=np.mean([AMI([a[y][k] for k in a[y]],[b[y][k] for k in a[y]]) for y in a])))
print(pd.DataFrame(ref).groupby('ds').ami.mean().round(3))
# what are the top ASCENDANT/RESIDUAL in V3 seed0
for ds in ['ACM','DBLP']:
    df=V[(ds,'V3',0)]['df']
    print(ds,'V3 top by mean_cvs'); print(df.sort_values('mean_cvs',ascending=False).head(5)[['mean_cvs','cmr','lifespan','zone','label']].to_string())
