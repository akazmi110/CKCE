import pickle, numpy as np, pandas as pd, collections, random
from common import *; from cvta import *
R=pickle.load(open('leiden_0_50.pkl','rb'))
out=[]; trk=[]; vp=[]
for ds in ['ACM','DBLP']:
    G=build_graphs(load(ds)); V={y:set(G[y].nodes()) for y in G}
    for y in range(2004,2024): vp.append(dict(ds=ds,year=y,persist=len(V[y]&V[y+1])/len(V[y])))
    for seed in range(50):
        A=R[(ds,seed)]; cvs=compute_cvs(G,A)
        ch,ks,links=build_chains(A); df=chain_metrics(ch,cvs); df,tau=zones(df)
        tg=collections.Counter(t for t,_ in links.values())
        trk.append(dict(ds=ds,seed=seed,variant='greedy θ=0 (baseline)',links=len(links),many_to_one=sum(1 for c in tg.values() if c>=2),
            links_into_shared=sum(c for c in tg.values() if c>=2)/len(links), n=len(df),pos=np.mean(df.cmr>=0),
            **df.zone.value_counts(normalize=True).to_dict()))
        base_zone={tuple(n):z for n,z in zip(df.nodes,df.zone)}
        for name,kw in [('mutual best match',dict(mutual=True)),('greedy θ=0.05',dict(theta=0.05)),('greedy θ=0.10',dict(theta=0.10))]:
            ch2,_,l2=build_chains(A,**kw); d2=chain_metrics(ch2,cvs)
            if len(d2)==0: trk.append(dict(ds=ds,seed=seed,variant=name,links=len(l2),n=0)); continue
            d2,_=zones(d2,tau)  # same threshold for comparability
            # zone agreement for chains sharing the same terminal community
            bterm={n[-1]:z for n,z in zip(df.nodes,df.zone)}
            same=[bterm[n[-1]]==z for n,z in zip(d2.nodes,d2.zone) if n[-1] in bterm]
            trk.append(dict(ds=ds,seed=seed,variant=name,links=len(l2),n=len(d2),pos=np.mean(d2.cmr>=0),zone_agree=np.mean(same) if same else np.nan,
                **d2.zone.value_counts(normalize=True).to_dict()))
        for _,r in df[df.lifespan>=5].iterrows():
            n0=r.nodes[0]; s0=ks[n0]
            for k,n in enumerate(r.nodes):
                s=ks[n]; j=len(s0&s)/len(s0|s); ret=len(s0&s)/len(s0)
                # baseline: random keyword set of the same size drawn from year vocabulary
                rng=random.Random(hash((seed,n))%10**6); voc=list(V[n[0]]); rs=set(rng.sample(voc,min(len(s),len(voc))))
                out.append(dict(ds=ds,seed=seed,chain=r.chain_id,step=k,span=n[0]-n0[0],J0=j,ret0=ret,Jrand=len(s0&rs)/len(s0|rs),retrand=len(s0&rs)/len(s0),life=r.lifespan,zone=r.zone))
D=pd.DataFrame(out); T=pd.DataFrame(trk); VP=pd.DataFrame(vp)
D.to_csv('drift.csv',index=False); T.to_csv('tracking.csv',index=False); VP.to_csv('vocab_persist.csv',index=False)
print(VP.groupby('ds').persist.describe().round(3))
print(T.groupby(['ds','variant']).mean(numeric_only=True).round(3).drop(columns='seed').to_string())
D['span_b']=pd.cut(D.span,[-1,0,2,5,10,20])
print(D.groupby(['ds','span_b']).agg(J0=('J0','mean'),ret0=('ret0','mean'),Jr=('Jrand','mean'),retr=('retrand','mean'),n=('J0','size')).round(3))
