import json, numpy as np, collections
from common import *
A={p['paper_id']:p for p in load('ACM')}; D={p['paper_id']:p for p in load('DBLP')}
both=[i for i in A if i in D]
same=[i for i in both if A[i]['original_keywords']==D[i]['original_keywords']]
J=[];exact=0;ov=[]
for i in same:
    a=set(k.strip().lower() for k in A[i]['cleaned_keywords']); d=set(k.strip().lower() for k in D[i]['cleaned_keywords'])
    j=len(a&d)/len(a|d) if a|d else 1; J.append(j); exact+=(a==d)
J=np.array(J)
print('shared ids',len(both),'identical input',len(same),'exact output match %.1f%%'%(100*exact/len(same)),'mean J %.3f median %.3f'%(J.mean(),np.median(J)), 'J>=0.8 %.1f%%'%(100*np.mean(J>=0.8)))
# per-keyword consistency: for an original keyword occurring >=3 times within a dataset, is it kept identically?
for ds in ['ACM','DBLP']:
    P=load(ds); occ=collections.defaultdict(list)
    for p in P:
        C=set(k.strip().lower() for k in p['cleaned_keywords'])
        for o in p['original_keywords']:
            o2=o.strip().lower(); occ[o2].append(o2 in C)
    m=[(k,np.mean(v)) for k,v in occ.items() if len(v)>=3]
    # consistency = share of occurrences agreeing with the majority decision (kept verbatim vs transformed)
    cons=[max(np.mean(occ[k]),1-np.mean(occ[k])) for k,_ in m]
    w=[len(occ[k]) for k,_ in m]
    print(ds,'orig keywords with >=3 occurrences',len(m),'mean decision consistency %.3f weighted %.3f'%(np.mean(cons),np.average(cons,weights=w)), 'fully consistent %.1f%%'%(100*np.mean(np.array(cons)==1)))
json.dump(dict(shared=len(both),same=len(same),exact=exact,meanJ=J.mean(),medJ=float(np.median(J)),ge08=float(np.mean(J>=0.8))),open('retest.json','w'))
