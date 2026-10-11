import re, pickle, collections, numpy as np, sys
from nltk.stem import PorterStemmer
from common import *; from cvta import *
ps=PorterStemmer()
def rule(k): k=k.lower().strip(); k=re.sub(r"[^\w\s]|_"," ",k); return re.sub(r"\s+"," ",k).strip()
def squash(k): return ''.join(ps.stem(t) for t in rule(k).split())
def canon_map(P,key):
    cnt=collections.Counter(k.strip().lower() for p in P for k in (p[key] or []) if k and k.strip())
    best={}
    for k,c in cnt.most_common():
        best.setdefault(squash(k),k)
    return {k:best[squash(k)] for k in cnt}
HUB=re.compile(r'^(cyber ?-?security|cyber ?-?secur|cyber|security|secure|cyber ?sec)$')
def variant_papers(P,v):
    out=[]
    if v=='V1':
        cm=canon_map(P,'cleaned_keywords')
        for p in P: q=dict(p); q['kw']=[cm[k.strip().lower()] for k in p['cleaned_keywords'] if k and k.strip()]; out.append(q)
    elif v=='V2':
        P2=[dict(p, ruled=[rule(k) for k in p['original_keywords'] if rule(k)]) for p in P]
        cm=canon_map(P2,'ruled')
        for p in P2: p['kw']=[cm[k] for k in p['ruled']]; out.append(p)
    elif v=='V3':
        for p in P: q=dict(p); q['kw']=[k for k in p['cleaned_keywords'] if k and not HUB.match(k.strip().lower())]; out.append(q)
    else:
        for p in P: q=dict(p); q['kw']=p['cleaned_keywords']; out.append(q)
    return out
if __name__=='__main__':
    res={}
    for ds in ['ACM','DBLP']:
        P=load(ds)
        for v in sys.argv[1:]:
            G=build_graphs(variant_papers(P,v),key='kw'); H={y:to_ig(g) for y,g in G.items()}
            for seed in range(10):
                A={y:leiden(G[y],seed,H[y]) for y in sorted(G)}
                cvs=compute_cvs(G,A); ch,_,_=build_chains(A); df=chain_metrics(ch,cvs); df,tau=zones(df)
                df['label']=[label(n[-1],A,G,10) for n in df.nodes]
                res[(ds,v,seed)]=dict(A=A,df=df,tau=tau)
                print(ds,v,seed,len(df),round(tau,4),df.zone.value_counts().to_dict(),flush=True)
            res[(ds,v,'nodes')]={y:set(G[y].nodes()) for y in G}
    pickle.dump(res,open('variants_%s.pkl'%'_'.join(sys.argv[1:]),'wb'))
