import json, itertools, collections, networkx as nx
DATA='../data'
FILES={'ACM':DATA+'/acm_cyber_security_keywords.openai-gpt.jsonl','DBLP':DATA+'/dblp_cleaned_multi_free.openai-gpt.jsonl'}
RAW={'ACM':DATA+'/acm_cyber_security_regex.jsonl','DBLP':DATA+'/dblp_cyber_security_regex.jsonl'}
def load(ds):
    return [json.loads(l) for l in open(FILES[ds],encoding='utf-8') if l.strip()]
def build_graphs(papers, key='cleaned_keywords', y0=2004, y1=2024, drop=None):
    pc=collections.defaultdict(lambda: collections.defaultdict(int))
    for p in papers:
        y=p['year']
        if not isinstance(y,int) or y<y0 or y>y1: continue
        kws=sorted(set(k.strip().lower() for k in (p.get(key) or []) if k and k.strip()))
        if drop: kws=[k for k in kws if not drop(k)]
        for a,b in itertools.combinations(kws,2):
            pc[y][(a,b)]+=1
    G={}
    for y,d in pc.items():
        g=nx.Graph()
        for (a,b),w in d.items(): g.add_edge(a,b,weight=w)
        G[y]=g
    return G
