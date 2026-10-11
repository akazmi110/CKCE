import pickle, numpy as np, pandas as pd, collections
from common import *; from cvta import *
R=pickle.load(open('leiden_0_50.pkl','rb')); rows=[]
for ds in ['ACM','DBLP']:
    G=build_graphs(load(ds)); V={y:set(G[y].nodes()) for y in G}
    for seed in range(10):
        A=R[(ds,seed)]; cvs=compute_cvs(G,A); ch,ks,links=build_chains(A); df=chain_metrics(ch,cvs)
        for src,(tgt,j) in links.items():
            a,b=ks[src],ks[tgt]; a2,b2=a&V[tgt[0]],b&V[src[0]]
            js=len(a&b)/len(a2|b2) if a2|b2 else np.nan
            rows.append(dict(ds=ds,seed=seed,kind='adjacent',J=j,Jstar=js))
        for _,r in df[df.lifespan>=6].iterrows():
            n0=r.nodes[0]; n=r.nodes[-1]; a,b=ks[n0],ks[n]; a2,b2=a&V[n[0]],b&V[n0[0]]
            rows.append(dict(ds=ds,seed=seed,kind='first-to-last (lifespan>=6)',J=len(a&b)/len(a|b),Jstar=len(a&b)/len(a2|b2) if a2|b2 else np.nan))
D=pd.DataFrame(rows); print(D.groupby(['ds','kind'])[['J','Jstar']].describe().round(3).T.to_string())
D.to_csv('jstar.csv',index=False)
