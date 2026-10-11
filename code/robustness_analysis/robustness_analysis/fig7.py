import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig=plt.figure(figsize=(16,7.2)); ax=fig.add_axes([0.10,0.16,0.37,0.74])
tau=0.35; xmax=1.0; ymin,ymax=-1,1
cols={'IGN':'#fde9d4','ASC':'#e2f0dd','RES':'#dce8f8','PER':'#ece4f0'}
ax.add_patch(Rectangle((0,0),tau,ymax,color=cols['IGN'])); ax.add_patch(Rectangle((tau,0),xmax-tau,ymax,color=cols['ASC']))
ax.add_patch(Rectangle((tau,ymin),xmax-tau,-ymin,color=cols['RES'])); ax.add_patch(Rectangle((0,ymin),tau,-ymin,color=cols['PER']))
ax.axhline(0,color='k',lw=2); ax.axvline(tau,color='k',lw=2,ls='--')
ax.set_xlim(0,xmax); ax.set_ylim(ymin,ymax)
ax.set_xticks([0,tau]); ax.set_xticklabels(['0',r'$\tau_{\mathrm{CVS}}$' '\n(median of ' r'$\overline{\mathrm{CVS}}$)'])
ax.set_yticks([ymin*0.75,0,ymax*0.75]); ax.set_yticklabels(['net decline\n(CMR < 0)','0','net gain\n(CMR > 0)'])
ax.set_xlabel(r'Mean vitality $\overline{\mathrm{CVS}}(\Lambda)$  (share of yearly co-occurrence weight, $\geq 0$)',fontsize=12)
ax.set_ylabel(r'Compound Momentum Rate $\mathrm{CMR}(\Lambda)$ (endpoint-to-endpoint)',fontsize=12)
bb=dict(boxstyle='round,pad=0.4',fc='white',ec='k',lw=1.2)
for x,y,t in [(tau/2,0.55,'IGNITION\nlow mean vitality,\nnet gain'),((tau+xmax)/2,0.55,'ASCENDANT\nhigh mean vitality,\nnet gain'),
              ((tau+xmax)/2,-0.55,'RESIDUAL\nhigh mean vitality,\nnet decline'),(tau/2,-0.55,'PERIPHERAL\nlow mean vitality,\nnet decline')]:
    ax.text(x,y,t,ha='center',va='center',bbox=bb,fontsize=12)
ax.set_title('Thematic Trajectory Map (TTMap) — schematic layout',fontsize=14,fontweight='bold')
ax.text(0.99,-0.97,'Schematic; positions not to scale.',ha='right',va='bottom',fontsize=9.5,style='italic')
# table
tx=fig.add_axes([0.52,0.10,0.46,0.80]); tx.axis('off')
rows=[('Zone',r'$\overline{\mathrm{CVS}}$','CMR','Interpretation'),
 ('IGNITION',r'$<\tau_{\mathrm{CVS}}$',r'$\geq 0$','Below-median average share of co-occurrence\nweight; final-year CVS $\\geq$ first-year CVS.\nCandidate emerging theme.'),
 ('ASCENDANT',r'$\geq\tau_{\mathrm{CVS}}$',r'$\geq 0$','Above-median average share; final-year CVS\n$\\geq$ first-year CVS. Prominent theme with\nnet relative gain over its observed span.'),
 ('RESIDUAL',r'$\geq\tau_{\mathrm{CVS}}$',r'$< 0$','Above-median average share; final-year CVS\n< first-year CVS. Prominent theme with net\nrelative loss over its observed span.'),
 ('PERIPHERAL',r'$<\tau_{\mathrm{CVS}}$',r'$< 0$','Below-median average share; final-year CVS\n< first-year CVS. Minor theme with net\nrelative loss over its observed span.')]
fills=['#2c3e50',cols['IGN'],cols['ASC'],cols['RES'],cols['PER']]
widths=[0.18,0.12,0.09,0.61]; h=[0.12,0.21,0.21,0.21,0.21]; y=1.0
for r,(row,fc) in enumerate(zip(rows,fills)):
    y-=h[r]; x=0
    for c,(txt,w) in enumerate(zip(row,widths)):
        f=fc if (r==0 or c==0) else ('white' if r%2 else '#f7f8fa')
        tx.add_patch(Rectangle((x,y),w,h[r],fc=f,ec='#9aa3ad',lw=1.2))
        tx.text(x+(0.015 if c==3 and r>0 else w/2),y+h[r]/2,txt,ha='left' if c==3 and r>0 else 'center',va='center',
                fontsize=12 if r==0 or c<3 else 11, color='white' if r==0 else 'black', fontweight='bold' if (r==0 or c==0) else 'normal')
        x+=w
tx.set_xlim(0,1); tx.set_ylim(0,1)
tx.set_title('Trajectory-zone definitions',fontsize=14,fontweight='bold')
fig.savefig('./ttmap_quadrant_table.png',dpi=200)
