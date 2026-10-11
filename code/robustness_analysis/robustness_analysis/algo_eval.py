import pickle, numpy as np, pandas as pd, collections, itertools, math, random, json
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score as AMI
from common import *
R=pickle.load(open('algos.pkl','rb'))
ALGS=['Leiden','Louvain','Infomap','Walktrap','LabelProp']
def groups(a):
    g=collections.defaultdict(set)
    for k,c in a.items(): g[c].add(k)
    return list(g.values())
def best_jacc(a,b):
    ga,gb=groups(a),groups(b); inv=collections.defaultdict(list)
    for i,s in enumerate(gb):
        for k in s: inv[k].append(i)
    out=[]
    for s in ga:
        cnt=collections.Counter(i for k in s for i in inv.get(k,()))
        out.append(max([cnt[i]/len(s|gb[i]) for i in cnt],default=0.0))
    return np.array(out)
def shuffle_part(a,rng):
    keys=list(a); labs=[a[k] for k in keys]; rng.shuffle(labs); return dict(zip(keys,labs))
def events(a,b,delta=0.3,minsurv=3):
    ga,gb=groups(a),groups(b); vb=set(b); lab_b=b
    succ=collections.defaultdict(list); pred=collections.defaultdict(list); deaths=0; nfor=0
    for i,s in enumerate(ga):
        S=s&vb
        if len(S)<minsurv: deaths+=1; continue
        cnt=collections.Counter(lab_b[k] for k in S)
        ss=[c for c,n in cnt.items() if n/len(S)>=delta]
        if not ss: deaths+=1; continue
        for c in ss: succ[i].append(c); pred[c].append(i)
    splits=sum(1 for i in succ if len(succ[i])>=2)
    merges=sum(1 for c in pred if len(pred[c])>=2)
    cont=sum(1 for i in succ if len(succ[i])==1 and len(pred[succ[i][0]])==1)
    births=len(set(lab_b.values())-set(pred))
    return dict(n_t=len(ga),cont=cont,split=splits,merge=merges,death=deaths,birth=births,n_t1=len(gb))
rows=[]; ev=[]; tj=[]
for ds in ['ACM','DBLP']:
    P=load(ds); G=build_graphs(P)
    # document sets per year for leave-year-out NPMI
    docs=collections.defaultdict(list)
    for p in P:
        if isinstance(p['year'],int) and 2004<=p['year']<=2024:
            docs[p['year']].append(set(k.strip().lower() for k in p['cleaned_keywords'] if k and k.strip()))
    alldf=collections.Counter(); N=0
    for y in docs:
        for d in docs[y]: alldf.update(d); N+=1
    def npmi_ref(y):
        # leave-year-out document frequencies
        df1=alldf.copy(); n=N-len(docs[y])
        for d in docs[y]: df1.subtract(d)
        return df1,n
    ref={y:npmi_ref(y) for y in G}
    # pair df computed lazily via inverted index (excluding year y)
    inv=collections.defaultdict(set); docyear=[]
    i=0
    for y in sorted(docs):
        for d in docs[y]:
            for k in d: inv[k].add(i)
            docyear.append(y); i+=1
    docyear=np.array(docyear)
    def npmi(words,y):
        df1,n=ref[y]; vals=[]
        for a,b in itertools.combinations(words,2):
            pa,pb=df1[a]/n,df1[b]/n
            if pa==0 or pb==0: continue
            co=sum(1 for d in inv[a]&inv[b] if docyear[d]!=y)/n
            if co==0: vals.append(-1.0); continue
            pmi=math.log(co/(pa*pb)); vals.append(pmi/(-math.log(co)) if co<1 else 1.0)
        return np.mean(vals) if vals else np.nan
    for alg in ALGS:
        seeds=[k[2] for k in R if k[0]==ds and k[1]==alg]
        for s in seeds:
            A=R[(ds,alg,s)]
            for y in sorted(G):
                g=G[y]; a=A[y]; comms=groups(a)
                Q=nx.community.modularity(g,comms,weight='weight')
                tw=sum(d['weight'] for *_,d in g.edges(data=True))
                cov=sum(d['weight'] for u,v,d in g.edges(data=True) if a[u]==a[v])/tw
                disc=sum(1 for c in comms if len(c)>1 and not nx.is_connected(g.subgraph(c)))
                row=dict(ds=ds,alg=alg,seed=s,year=y,Q=Q,cov=cov,k=len(comms),disc=disc/len(comms))
                if s==0:
                    co=[]; 
                    for c in comms:
                        if len(c)<5: continue
                        sg=g.subgraph(c); st=dict(sg.degree(weight='weight')); top=sorted(st,key=st.get,reverse=True)[:10]
                        co.append((npmi(top,y),len(c)))
                    co=[x for x in co if not np.isnan(x[0])]
                    row['npmi']=np.mean([x for x,_ in co]) if co else np.nan
                    row['npmi_w']=np.average([x for x,_ in co],weights=[w for _,w in co]) if co else np.nan
                rows.append(row)
            years=sorted(G)
            for y,y1 in zip(years[:-1],years[1:]):
                bj=best_jacc(A[y],A[y1])
                rng=random.Random(s*1000+y); base=[best_jacc(shuffle_part(A[y],rng),shuffle_part(A[y1],rng)).mean() for _ in range(20)]
                tj.append(dict(ds=ds,alg=alg,seed=s,year=y,obs=bj.mean(),base=np.mean(base),base_sd=np.std(base)))
                if s==0: ev.append(dict(ds=ds,alg=alg,year=y,**events(A[y],A[y1])))
        # seed stability: pairwise AMI between seeds per year
        if len(seeds)>1:
            for y in sorted(G):
                v=[]
                for s1,s2 in itertools.combinations(seeds,2):
                    a1,a2=R[(ds,alg,s1)][y],R[(ds,alg,s2)][y]; ks=list(a1)
                    v.append(AMI([a1[k] for k in ks],[a2[k] for k in ks]))
                rows.append(dict(ds=ds,alg=alg,seed=-1,year=y,seedAMI=np.mean(v)))
        print(ds,alg,'done',flush=True)
pd.DataFrame(rows).to_csv('algo_rows.csv',index=False); pd.DataFrame(ev).to_csv('events.csv',index=False); pd.DataFrame(tj).to_csv('tj.csv',index=False)
