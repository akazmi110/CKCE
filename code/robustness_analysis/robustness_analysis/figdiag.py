import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
B,O,G='#2a78d6','#eb6834','#9a9993'
SH=pd.read_csv('shape.csv'); D=pd.read_csv('drift.csv'); J=pd.read_csv('jstar_base.csv')
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
# (a) CMR vs log slope
a=ax[0]; agree=np.sign(SH.cmr)==np.sign(SH.slope)
a.scatter(SH.slope[agree],SH.cmr[agree],s=10,c=B,alpha=.35,lw=0,label=f'same sign ({agree.mean()*100:.1f}%)')
a.scatter(SH.slope[~agree],SH.cmr[~agree],s=14,c=O,alpha=.9,lw=0,label=f'opposite sign ({(~agree).mean()*100:.1f}%)')
a.axhline(0,color=G,lw=1); a.axvline(0,color=G,lw=1); a.set_yscale('symlog',linthresh=1)
a.set_xlabel('OLS slope of ln CVS on year (trend)'); a.set_ylabel('CMR (endpoint-to-endpoint, symlog)')
a.set_title('(a) CMR vs. log-linear trend',loc='left',fontweight='bold'); a.legend(frameon=False,loc='upper left')
# (b) drift
b=ax[1]; D=D[D.span>0]
g=D.groupby('span').agg(J0=('J0','mean'),Jr=('Jrand','mean'),n=('J0','size')); g=g[g.n>=30]
b.plot(g.index,g.J0,color=B,lw=2,marker='o',ms=5,label='chain community vs. its first community')
b.plot(g.index,g.Jr,color=G,lw=2,ls='--',marker='o',ms=5,label='random same-size keyword set')
b.set_xlabel('Years since chain start'); b.set_ylabel('Jaccard with first community'); b.set_ylim(0,None)
b.set_title('(b) Cumulative drift along chains (lifespan ≥ 5)',loc='left',fontweight='bold'); b.legend(frameon=False)
# (c) persistent-vocabulary J vs baseline
c=ax[2]; s=J.groupby('alg')[['obs','base']].mean().reindex(['Leiden','Louvain','Infomap','Walktrap','LabelProp'])
x=np.arange(len(s)); w=.38
c.bar(x-w/2-0.01,s.obs,w,color=B,label='observed'); c.bar(x+w/2+0.01,s.base,w,color=G,label='size-preserving shuffle')
c.set_xticks(x); c.set_xticklabels(['Leiden','Louvain','Infomap','Walktrap','Label\nProp.']); c.set_ylabel('Mean best-match Jaccard*\n(persistent vocabulary)')
c.set_title('(c) Continuity vs. random baseline',loc='left',fontweight='bold'); c.set_ylim(0,0.46); c.legend(frameon=False,ncol=2,loc='upper left')
for i,(o,bb) in enumerate(zip(s.obs,s.base)): c.text(i,max(o,bb)+0.008,f'×{o/bb:.2f}',ha='center',fontsize=9,color='#52514e')
fig.tight_layout(); fig.savefig('./robustness_diagnostics.png',dpi=220)
