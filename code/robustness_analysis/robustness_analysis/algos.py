import pickle, sys, time, random, numpy as np, networkx as nx, igraph as ig, leidenalg as la, infomap
from common import *; from cvta import to_ig
def part_from(h, comms): return {h.vs[i]['name']:c for c,cm in enumerate(comms) for i in cm}
def run(alg, g, h, seed):
    if alg=='Leiden': return part_from(h, la.find_partition(h, la.ModularityVertexPartition, weights='weight', seed=seed))
    if alg=='Louvain': return {n:c for c,cm in enumerate(nx.community.louvain_communities(g, weight="weight", seed=seed)) for n in cm}
    if alg=='Infomap':
        im=infomap.Infomap(silent=True, seed=seed+1, two_level=True)
        nodes=list(g.nodes()); idx={n:i for i,n in enumerate(nodes)}
        for u,v,d in g.edges(data=True): im.add_link(idx[u],idx[v],d['weight'])
        im.run(); return {nodes[k]:v for k,v in im.get_modules().items()}
    if alg=='Walktrap': return part_from(h, h.community_walktrap(steps=4).as_clustering())   # unweighted, as in released code
    if alg=='LabelProp':
        return {n:c for c,cm in enumerate(nx.algorithms.community.label_propagation_communities(g)) for n in cm}  # unweighted, as in released code
ALGS={'Leiden':10,'Louvain':10,'Infomap':10,'Walktrap':1,'LabelProp':1}
out={}
for ds in ['ACM','DBLP']:
    G=build_graphs(load(ds)); H={y:to_ig(g) for y,g in G.items()}
    for alg,ns in ALGS.items():
        t=time.time()
        for s in range(ns):
            out[(ds,alg,s)]={y:run(alg,G[y],H[y],s) for y in sorted(G)}
        print(ds,alg,ns,'runs %.1fs'%(time.time()-t),flush=True)
pickle.dump(out,open('algos.pkl','wb'))
