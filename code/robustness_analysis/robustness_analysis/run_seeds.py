import pickle, sys
from common import *; from cvta import *
res={}
for ds in ['ACM','DBLP']:
    G=build_graphs(load(ds)); H={y:to_ig(g) for y,g in G.items()}
    for seed in range(int(sys.argv[1]),int(sys.argv[2])):
        A={y:leiden(G[y],seed,H[y]) for y in sorted(G)}
        res[(ds,seed)]=A
        cvs=compute_cvs(G,A); ch,_,_=build_chains(A); df=chain_metrics(ch,cvs); df,tau=zones(df)
        print(ds,seed,len(df),round(tau,4),df.zone.value_counts().to_dict(), round(df.cmr.mean(),3), flush=True)
pickle.dump(res,open(f'leiden_{sys.argv[1]}_{sys.argv[2]}.pkl','wb'))
