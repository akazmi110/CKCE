import numpy as np, pandas as pd, igraph as ig, leidenalg as la, collections
def to_ig(g):
    nodes=list(g.nodes()); idx={n:i for i,n in enumerate(nodes)}
    h=ig.Graph(n=len(nodes), edges=[(idx[u],idx[v]) for u,v in g.edges()])
    h.es['weight']=[d['weight'] for _,_,d in g.edges(data=True)]
    h.vs['name']=nodes
    return h
def leiden(g, seed=None, h=None):
    h=h or to_ig(g)
    p=la.find_partition(h, la.ModularityVertexPartition, weights='weight', seed=seed)
    return {h.vs[i]['name']:c for c,comm in enumerate(p) for i in comm}
def louvain(g, seed=None, h=None):
    import random; random.seed(seed); 
    h=h or to_ig(g); ig.set_random_number_generator(random)
    p=h.community_multilevel(weights='weight')
    return {h.vs[i]['name']:c for c,comm in enumerate(p) for i in comm}
def compute_cvs(G, A):
    rec={}
    for y,g in G.items():
        if y not in A: continue
        a=A[y]; tw=sum(d['weight'] for _,_,d in g.edges(data=True)); vol=collections.Counter()
        for u,v,d in g.edges(data=True):
            if a.get(u) is not None and a.get(u)==a.get(v): vol[a[u]]+=d['weight']
        for c,vv in vol.items(): rec[(y,c)]=vv/tw
    return rec
def kwsets(A):
    ks=collections.defaultdict(set)
    for y,a in A.items():
        for k,c in a.items(): ks[(y,c)].add(k)
    return ks
def forward_links(A, theta=0.0, mutual=False):
    ks=kwsets(A); years=sorted(A); links={}
    by=collections.defaultdict(list)
    for (y,c) in ks: by[y].append(c)
    best_back={}
    for i in range(len(years)-1):
        y,y1=years[i],years[i+1]
        # inverted index for speed
        inv=collections.defaultdict(set)
        for c1 in by[y1]:
            for k in ks[(y1,c1)]: inv[k].add(c1)
        for c in by[y]:
            s=ks[(y,c)]; cnt=collections.Counter()
            for k in s:
                for c1 in inv.get(k,()): cnt[c1]+=1
            bj,bc=0.0,None
            for c1 in sorted(cnt):
                j=cnt[c1]/len(s|ks[(y1,c1)])
                if j>bj: bj,bc=j,c1
            if bc is not None and bj>theta: links[(y,c)]=((y1,bc),bj)
    if mutual:
        # keep only links where source is also the best predecessor of target
        bestpred={}
        for src,(tgt,j) in links.items():
            if tgt not in bestpred or j>bestpred[tgt][1]: bestpred[tgt]=(src,j)
        links={s:(t,j) for s,(t,j) in links.items() if bestpred[t][0]==s}
    return links, ks
def build_chains(A, theta=0.0, mutual=False):
    links,ks=forward_links(A,theta,mutual)
    visited=set(); chains=[]
    for node in sorted(ks):
        if node in visited: continue
        ch=[node]; visited.add(node); cur=node
        while cur in links:
            nxt=links[cur][0]
            if nxt in visited: break
            ch.append(nxt); visited.add(nxt); cur=nxt
        chains.append(ch)
    return chains, ks, links
def chain_metrics(chains, cvs, lmin=3):
    out=[]
    for cid,ch in enumerate(chains):
        if len(ch)<lmin: continue
        vals=[(n,cvs.get(n,0.0)) for n in ch]; valid=[(n,v) for n,v in vals if v>0]
        if len(valid)<lmin: continue
        v=np.array([x for _,x in valid]); yrs=np.array([n[0] for n,_ in valid])
        ratios=v[1:]/v[:-1]
        cmr=np.prod(ratios)**(1/len(ratios))-1
        slope=np.polyfit(yrs-yrs[0], np.log(v),1)[0]
        frac_up=np.mean(ratios>1)
        out.append(dict(chain_id=cid, mean_cvs=v.mean(), cmr=cmr, lifespan=len(valid),
            first_year=int(yrs[0]), last_year=int(yrs[-1]), cvs_first=v[0], cvs_last=v[-1],
            log_slope=slope, frac_up=frac_up, nodes=[n for n,_ in valid]))
    return pd.DataFrame(out)
def zones(df, tau=None):
    tau=df.mean_cvs.median() if tau is None else tau
    def z(r):
        h=r.mean_cvs>=tau; p=r.cmr>=0
        return ('ASCENDANT' if h else 'IGNITION') if p else ('RESIDUAL' if h else 'PERIPHERAL')
    df=df.copy(); df['zone']=df.apply(z,axis=1); return df,tau
def label(node, A, G, k=5):
    y,c=node; g=G[y]; mem={n for n,cc in A[y].items() if cc==c}
    deg={n:0.0 for n in mem}
    for u,v,d in g.subgraph(mem).edges(data=True): deg[u]+=d['weight']; deg[v]+=d['weight']
    return ', '.join(sorted(deg,key=deg.get,reverse=True)[:k])
