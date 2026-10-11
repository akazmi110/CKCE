import pandas as pd, numpy as np
pd.set_option('display.width',250)
R=pd.read_csv('algo_rows.csv'); T=pd.read_csv('tj.csv'); E=pd.read_csv('events.csv')
q=R[R.seed>=0].groupby(['ds','alg']).agg(Q=('Q','mean'),cov=('cov','mean'),k=('k','mean'),disc=('disc','mean'),npmi=('npmi','mean'),npmi_w=('npmi_w','mean'))
s=R[R.seed==-1].groupby(['ds','alg']).seedAMI.mean()
T['ratio']=T.obs/T.base; T['z']=(T.obs-T.base)/T.base_sd.replace(0,np.nan)
t=T.groupby(['ds','alg']).agg(Jobs=('obs','mean'),Jbase=('base','mean'),ratio=('ratio','mean'),z=('z','median'),
     Jobs_sd_seed=('obs',lambda x: np.nan))
# seed sd of the period-mean Jaccard
ts=T.groupby(['ds','alg','seed']).obs.mean().groupby(['ds','alg']).std()
t['Jobs_sd_seed']=ts
out=q.join(s).join(t); print(out.round(3).to_string()); out.to_csv('algo_table.csv')
E['phase']=pd.cut(E.year,[2003,2010,2017,2024],labels=['I','II','III'])
e=E.groupby(['ds','alg']).agg(cont=('cont','sum'),split=('split','sum'),merge=('merge','sum'),death=('death','sum'),birth=('birth','sum'),n=('n_t','sum'))
print(e.to_string()); e.to_csv('events_table.csv')
print(E[E.alg=='Leiden'].groupby(['ds','phase']).agg(cont=('cont','sum'),split=('split','sum'),merge=('merge','sum'),death=('death','sum'),birth=('birth','sum'),n=('n_t','sum')).to_string())
