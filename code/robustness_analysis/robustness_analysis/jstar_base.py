import pickle, numpy as np, pandas as pd, collections, random
from common import *
R=pickle.load(open('algos.pkl','rb')); rows=[]
def groups(a):
    g=collections.defaultdict(set)
    for k,c in a.items(): g[c].add(k)
    return list(g.values())
def bj_star(a,b):
    va,vb=set(a),set(b); ga=[s&vb for s in groups(a)]; gb=[s&va for s in groups(b)]
    inv=collections.defaultdict(list)
    for i,s in enumerate(gb):
        for k in s: inv[k].append(i)
    out=[]
    for s in ga:
        if not s: continue
        cnt=collections.Counter(i for k in s for i in inv.get(k,()))
        out.append(max([cnt[i]/len(s|gb[i]) for i in cnt],default=0.0))
    return np.mean(out)
def shuf(a,rng):
    k=list(a); l=[a[x] for x in k]; rng.shuffle(l); return dict(zip(k,l))
for ds in ['ACM','DBLP']:
    for alg in ['Leiden','Louvain','Infomap','Walktrap','LabelProp']:
        A=R[(ds,alg,0)]; ys=sorted(A)
        for y,y1 in zip(ys[:-1],ys[1:]):
            o=bj_star(A[y],A[y1]); rng=random.Random(y)
            b=[bj_star(shuf(A[y],rng),shuf(A[y1],rng)) for _ in range(20)]
            rows.append(dict(ds=ds,alg=alg,year=y,obs=o,base=np.mean(b),sd=np.std(b)))
D=pd.DataFrame(rows); D['ratio']=D.obs/D.base; D['z']=(D.obs-D.base)/D.sd
D.to_csv('jstar_base.csv',index=False)
print(D.groupby(['ds','alg'])[['obs','base','ratio','z']].mean().round(3))
