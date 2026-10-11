import re, json, collections, numpy as np
from nltk.stem import PorterStemmer
from common import *
ps=PorterStemmer(); SW={'of','and','for','the','in','on','to','a','an','as','with'}
def rule(k):
    k=k.lower().strip(); k=re.sub(r"[^\w\s]|_"," ",k); return re.sub(r"\s+"," ",k).strip()
def toks(k): return [ps.stem(t) for t in rule(k).split()]
def squash(k): return ''.join(toks(k))
def initials(phrase, keep_sw):
    ws=[w for w in rule(phrase).split() if keep_sw or w not in SW]; return ''.join(w[0] for w in ws if w)
def expands(c, O):
    cand={t.rstrip('s') for o in O for t in rule(o).split() if 2<=len(t)<=7 and t.isalnum()}
    cand|={rule(o).replace(' ','').rstrip('s') for o in O if len(rule(o).replace(' ',''))<=7}
    i1,i2=initials(c,False),initials(c,True)
    return any(a and (a==i1 or a==i2 or (len(a)>=3 and (i1.startswith(a) or i2.startswith(a)))) for a in cand)
def classify(c,O,otoks,orule,osq):
    if c in O: return 'unchanged'
    rc=rule(c)
    if rc in orule: return 'lexical'
    if squash(c) in osq: return 'lexical'      # hyphen/space joins, plural etc.
    tc=set(toks(c))
    if any(toks(c)==toks(o) for o in O): return 'morphological'
    if tc<=otoks: return 'subphrase'
    if expands(c,O): return 'acronym'
    # partial: remaining new tokens explained by squashing adjacent original tokens
    new=tc-otoks
    osq_all=''.join(otoks)
    if all(any(t in squash(o) for o in O) for t in new): return 'lexical'
    return 'ungrounded'
res={}; examples={}
for ds in ['ACM','DBLP']:
    P=load(ds); st=collections.Counter(); ung=[]
    for p in P:
        O=[k for k in p['original_keywords'] if k and k.strip()]; C=[k for k in p['cleaned_keywords'] if k and k.strip()]
        otoks=set(t for k in O for t in toks(k)); orule={rule(k) for k in O}; osq={squash(k) for k in O}
        for c in C:
            k=classify(c,O,otoks,orule,osq); st[k]+=1
            if k=='ungrounded': ung.append(dict(id=p['paper_id'],orig=O,out=c))
        n_drop=sum(1 for o in O if not (set(toks(o)) & set(t for k in C for t in toks(k))) and not any(expands(c,[o]) for c in C))
        st['__dropped_orig']+=n_drop; st['__n_orig']+=len(O); st['__n_clean']+=len(C)
        st['__papers_empty']+= (len(C)==0)
    res[ds]=dict(st); examples[ds]=ung
    tot=sum(v for k,v in st.items() if not k.startswith('__'))
    print(ds, {k:(v, round(100*v/tot,2)) for k,v in st.items() if not k.startswith('__')}, 'dropped orig %.2f%%'%(100*st['__dropped_orig']/st['__n_orig']), 'empty',st['__papers_empty'])
json.dump(dict(res=res,ex=examples),open('norm_eval2.json','w'))
# variant fragmentation in cleaned vocabulary vs raw vocabulary
for ds in ['ACM','DBLP']:
    P=load(ds)
    for key in ['original_keywords','cleaned_keywords']:
        vocab=collections.Counter(k.strip().lower() for p in P if isinstance(p['year'],int) and 2004<=p['year']<=2024 for k in p[key] if k and k.strip())
        groups=collections.defaultdict(list)
        for v in vocab: groups[squash(v)].append(v)
        multi=[g for g in groups.values() if len(g)>1]
        print(ds,key,'unique',len(vocab),'variant-groups',len(multi),'surface forms in them',sum(len(g) for g in multi),
              'occurrences',sum(vocab[x] for g in multi for x in g))
        if key=='cleaned_keywords':
            top=sorted(multi,key=lambda g:-sum(vocab[x] for x in g))[:8]
            for g in top: print('    ',[(x,vocab[x]) for x in g])
    print(ds,'single tokens cyber/security in cleaned:', sum('cyber' in p['cleaned_keywords'] for p in P), sum('security' in p['cleaned_keywords'] for p in P),
          ' orig:', sum('cyber' in p['original_keywords'] for p in P), sum('security' in p['original_keywords'] for p in P))
