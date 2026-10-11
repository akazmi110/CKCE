import pickle, numpy as np, pandas as pd, collections, itertools, math
from common import *
R=pickle.load(open('algos.pkl','rb')); rows=[]
for ds in ['ACM','DBLP']:
    P=load(ds); G=build_graphs(P)
    docs=[(p['year'],set(k.strip().lower() for k in p['cleaned_keywords'] if k and k.strip())) for p in P if isinstance(p['year'],int) and 2004<=p['year']<=2024]
    inv=collections.defaultdict(set); yr=np.array([y for y,_ in docs])
    for i,(y,d) in enumerate(docs):
        for k in d: inv[k].add(i)
    ny=collections.Counter(yr.tolist()); N=len(docs)
    def dfo(k,y): return sum(1 for i in inv[k] if yr[i]!=y)
    for alg in ['Leiden','Louvain','Infomap','Walktrap','LabelProp']:
        A=R[(ds,alg,0)]
        for y,g in G.items():
            n=N-ny[y]; gr=collections.defaultdict(set)
            for k,c in A[y].items(): gr[c].add(k)
            for c in gr.values():
                if len(c)<5: continue
                st=dict(g.subgraph(c).degree(weight='weight'))
                cand=[k for k in sorted(st,key=st.get,reverse=True) if dfo(k,y)>=5][:10]
                if len(cand)<3: continue
                vals=[]
                for a,b in itertools.combinations(cand,2):
                    pa,pb=dfo(a,y)/n,dfo(b,y)/n; co=sum(1 for i in inv[a]&inv[b] if yr[i]!=y)/n
                    vals.append(-1.0 if co==0 else math.log(co/(pa*pb))/(-math.log(co)))
                pos=[v for v in vals if v>-1]; rows.append(dict(ds=ds,alg=alg,year=y,size=len(c),npmi=np.mean(vals),support=len(pos)/len(vals),npmi_pos=np.mean(pos) if pos else np.nan))
        print(ds,alg,flush=True)
D=pd.DataFrame(rows); D.to_csv('coherence.csv',index=False)
f=lambda x: np.average(x.npmi,weights=x['size'])
print(D.groupby(['ds','alg']).agg(npmi=('npmi','mean'),support=('support','mean'),npmi_pos=('npmi_pos','mean'),n=('npmi','size')).join(D.groupby(['ds','alg']).apply(f).rename('npmi_w')).round(3))
