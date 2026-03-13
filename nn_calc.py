import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import io, base64, json

plt.rcParams.update({
    'figure.facecolor':'#111827','axes.facecolor':'#0a0e1a',
    'axes.edgecolor':'#1e2d45','axes.labelcolor':'#94a3b8',
    'xtick.color':'#94a3b8','ytick.color':'#94a3b8',
    'xtick.labelsize':11,'ytick.labelsize':11,
    'axes.labelsize':12,'axes.titlesize':13,
    'text.color':'#e2e8f0','grid.color':'#1e2d45',
    'grid.linewidth':0.6,'axes.grid':True,
    'legend.facecolor':'#111827','legend.edgecolor':'#1e2d45','legend.fontsize':10,
    'font.size':11,
})
TEAL='#00d4aa'; ORANGE='#ff6b35'; MUTED='#64748b'; TEXT='#e2e8f0'
LEAKY_ALPHA=0.01; ELU_ALPHA=1.0

# ── Funções de ativação ────────────────────────────────────────────────────────
def sigmoid(z):      return 1/(1+np.exp(-np.clip(z,-500,500)))
def d_sigmoid(z):    s=sigmoid(z); return s*(1-s)
def d_sig(a):        return a*(1-a)          # a partir da saída (camada output)

def relu(z):         return np.maximum(0,z)
def d_relu(z):       return (z>0).astype(float)

def tanh_fn(z):      return np.tanh(z)
def d_tanh(z):       return 1-np.tanh(z)**2

def leaky_relu(z):   return np.where(z>0,z,LEAKY_ALPHA*z)
def d_leaky_relu(z): return np.where(z>0,1.0,LEAKY_ALPHA)

def elu(z):
    zc=np.clip(z,-500,0)
    return np.where(z>0,z,ELU_ALPHA*(np.exp(zc)-1))
def d_elu(z):
    zc=np.clip(z,-500,0)
    return np.where(z>0,1.0,ELU_ALPHA*np.exp(zc))

def swish(z):        s=sigmoid(z); return z*s
def d_swish(z):      s=sigmoid(z); return s+z*s*(1-s)

ACT_FNS={
    'sigmoid':    (sigmoid,    d_sigmoid,    'Sigmoide',   TEAL),
    'relu':       (relu,       d_relu,       'ReLU',       ORANGE),
    'tanh':       (tanh_fn,    d_tanh,       'Tanh',       '#a855f7'),
    'leaky_relu': (leaky_relu, d_leaky_relu, 'Leaky ReLU', '#eab308'),
    'elu':        (elu,        d_elu,        'ELU',        '#ec4899'),
    'swish':      (swish,      d_swish,      'Swish',      '#3b82f6'),
}

def _b64(fig):
    buf=io.BytesIO()
    plt.savefig(buf,format='png',bbox_inches='tight',dpi=110)
    s='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
    plt.close(fig); return s

# ── Diagrama da rede ───────────────────────────────────────────────────────────
NP={'X':(0.5,0.62),'HA1':(1.9,0.85),'HA2':(1.9,0.38),
    'HB1':(3.3,0.85),'HB2':(3.3,0.38),'Y':(4.7,0.62)}
NC={'X':'#a07820','HA1':'#2060a0','HA2':'#2060a0',
    'HB1':'#206840','HB2':'#206840','Y':'#802020'}
R=0.13

def _base_ax():
    fig,ax=plt.subplots(figsize=(11,5))
    ax.set_xlim(0,5.4); ax.set_ylim(-0.05,1.18); ax.axis('off')
    for lx,lb in [(0.5,'Entrada X'),(1.9,'Oculta A'),(3.3,'Oculta B'),(4.7,'Saída ŷ')]:
        ax.text(lx,1.10,lb,ha='center',fontsize=11,color=MUTED,style='italic')
    return fig,ax

def _arrow(ax,src,dst,color,lbl,sgn):
    xs,ys=NP[src]; xe,ye=NP[dst]; dx,dy=xe-xs,ye-ys; d=np.sqrt(dx**2+dy**2)
    ax.annotate("",xy=(xe-(dx/d)*R,ye-(dy/d)*R),xytext=(xs+(dx/d)*R,ys+(dy/d)*R),
                arrowprops=dict(arrowstyle="->",color=color,lw=1.5,alpha=0.85))
    ax.text((xs+xe)/2,(ys+ye)/2+sgn*0.055,lbl,fontsize=8.5,ha='center',color=color,
            fontweight='bold',bbox=dict(boxstyle='round,pad=0.12',fc='#111827',ec='none',alpha=0.92))

def _nodes(ax,vals,delta=False,grad_mags=None):
    max_mag=max(grad_mags.values()) if grad_mags else 1.0
    if max_mag==0: max_mag=1.0
    for n,(px,py) in NP.items():
        if delta and grad_mags and n!='X':
            norm=np.clip(grad_mags.get(n,0)/max_mag,0,1)
            ec_color=mcolors.to_hex(mcm.RdYlGn(norm))
            lw=2.8
        else:
            ec_color='#556'; lw=1.5
        ax.add_artist(plt.Circle((px,py),R,color=NC[n],ec=ec_color,lw=lw,zorder=4))
        top=n if (not delta or n=='X') else 'δ'
        ax.text(px,py+0.030,top,ha='center',va='center',fontsize=9.5,
                fontweight='bold',zorder=5,color='#cbd5e1')
        vc=ORANGE if (delta and n!='X') else TEXT
        ax.text(px,py-0.055,f"{vals.get(n,0):.4f}",ha='center',va='center',
                fontsize=8.5,color=vc,zorder=5)
    # colorbar legend if gradient mode
    if delta and grad_mags:
        sm=mcm.ScalarMappable(cmap='RdYlGn',norm=mcolors.Normalize(0,max_mag))
        sm.set_array([])
        cbar=plt.colorbar(sm,ax=ax,shrink=0.4,pad=0.02,aspect=15)
        cbar.set_label('|δ|',color=MUTED,fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=MUTED,labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(),color=MUTED)

# ── Gráficos de ativação ───────────────────────────────────────────────────────
def plot_activation(act_key):
    fn,dfn,name,color=ACT_FNS[act_key]
    z=np.linspace(-4,4,300); y=fn(z); dy=dfn(z)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))

    axes[0].plot(z,np.clip(y,-2,5),color=color,lw=2.5)
    axes[0].axhline(0,color=MUTED,ls=':',lw=0.8)
    axes[0].axvline(0,color=MUTED,ls=':',lw=0.8)
    axes[0].set_title(f'{name}  —  saída f(z)',fontsize=12,color=TEXT)
    axes[0].set_xlabel('z'); axes[0].set_ylabel('f(z)')

    if act_key=='sigmoid':
        axes[0].axhline(0.5,color=MUTED,ls='--',lw=1,label='y=0.5')
        axes[0].axhline(1,color=MUTED,ls=':',lw=0.8,alpha=0.5,label='y=1')
        axes[0].set_ylim(-0.1,1.2); axes[0].legend()
    elif act_key=='relu':
        axes[0].fill_between(z,y,0,where=(z>0),alpha=0.15,color=color)
        axes[0].text(-2,0.4,'zona morta\n(saída=0)',color=ORANGE,fontsize=8,ha='center')
        axes[0].set_ylim(-0.5,4.5)
    elif act_key=='tanh':
        axes[0].axhline(1,color=MUTED,ls='--',lw=1,label='máx=1')
        axes[0].axhline(-1,color=MUTED,ls='--',lw=1,label='mín=-1')
        axes[0].fill_between(z,y,0,alpha=0.12,color=color)
        axes[0].set_ylim(-1.3,1.3); axes[0].legend()
    elif act_key=='leaky_relu':
        axes[0].fill_between(z,y,0,where=(z>0),alpha=0.15,color=color)
        axes[0].fill_between(z,y,0,where=(z<=0),alpha=0.3,color=ORANGE)
        axes[0].text(-2,-0.06,f'α={LEAKY_ALPHA}',color=ORANGE,fontsize=9,ha='center')
        axes[0].set_ylim(-0.15,4.5)
    elif act_key=='elu':
        axes[0].axhline(-ELU_ALPHA,color=MUTED,ls='--',lw=1,label=f'mín= −α={-ELU_ALPHA}')
        axes[0].fill_between(z,y,0,alpha=0.12,color=color)
        axes[0].set_ylim(-1.4,4.5); axes[0].legend()
    elif act_key=='swish':
        axes[0].fill_between(z,y,0,alpha=0.12,color=color)
        min_idx=np.argmin(y)
        axes[0].axvline(z[min_idx],color=ORANGE,ls='--',lw=1,
                       label=f'mín≈{y[min_idx]:.3f} em z={z[min_idx]:.2f}')
        axes[0].set_ylim(-0.5,4.5); axes[0].legend()

    axes[1].plot(z,np.clip(dy,-0.05,1.4),color=ORANGE,lw=2.5)
    axes[1].fill_between(z,np.clip(dy,-0.05,1.4),alpha=0.15,color=ORANGE)
    axes[1].axhline(0,color=MUTED,ls=':',lw=0.8)
    axes[1].set_title("Derivada  f'(z)",fontsize=12,color=TEXT)
    axes[1].set_xlabel('z'); axes[1].set_ylabel("f'(z)")

    if act_key=='sigmoid':
        axes[1].axhline(0.25,color=MUTED,ls='--',lw=1,label='máx=0.25'); axes[1].legend()
        axes[1].set_ylim(-0.02,0.30)
    elif act_key in ('relu','leaky_relu'):
        axes[1].axhline(1.0,color=MUTED,ls='--',lw=1,label="f'=1 (ativa)")
        axes[1].axhline(LEAKY_ALPHA if act_key=='leaky_relu' else 0,
                       color=ORANGE,ls='--',lw=1,
                       label=f"f'={LEAKY_ALPHA if act_key=='leaky_relu' else 0} (inativa)")
        axes[1].legend(); axes[1].set_ylim(-0.05,1.3)
    elif act_key=='tanh':
        axes[1].axhline(1.0,color=MUTED,ls='--',lw=1,label='máx=1.0'); axes[1].legend()
        axes[1].set_ylim(-0.05,1.2)
    elif act_key=='elu':
        axes[1].axhline(1.0,color=MUTED,ls='--',lw=1,label='máx=1 (z>0)')
        axes[1].legend(); axes[1].set_ylim(-0.05,1.2)
    elif act_key=='swish':
        axes[1].axhline(1.0,color=MUTED,ls='--',lw=0.8,alpha=0.5)
        axes[1].set_ylim(-0.1,1.3)

    plt.tight_layout(); return _b64(fig)

def plot_activation_comparison():
    z=np.linspace(-4,4,300)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    order=['sigmoid','relu','tanh','leaky_relu','elu','swish']
    for key in order:
        fn,dfn,name,color=ACT_FNS[key]
        y=fn(z); dy=dfn(z)
        axes[0].plot(z,np.clip(y,-1.3,4.5),color=color,lw=2,label=name,alpha=0.9)
        axes[1].plot(z,np.clip(dy,-0.05,1.3),color=color,lw=2,label=name,alpha=0.9)
    axes[0].axhline(0,color=MUTED,ls=':',lw=0.5); axes[0].axvline(0,color=MUTED,ls=':',lw=0.5)
    axes[0].set_title('Saída — todas as funções',fontsize=11,color=TEXT)
    axes[0].set_xlabel('z'); axes[0].legend(fontsize=8,ncol=2); axes[0].set_ylim(-1.4,4.8)
    axes[1].axhline(0,color=MUTED,ls=':',lw=0.5)
    axes[1].set_title("Derivada — todas as funções",fontsize=11,color=TEXT)
    axes[1].set_xlabel('z'); axes[1].legend(fontsize=8,ncol=2); axes[1].set_ylim(-0.06,1.35)
    plt.suptitle('Comparativo: Sigmoide · ReLU · Tanh · Leaky ReLU · ELU · Swish',
                 color=TEXT,fontsize=11)
    plt.tight_layout(); return _b64(fig)

# ── Diagramas forward / backprop ───────────────────────────────────────────────
def plot_forward(X,hA,hB,yp,W1,W2,W3,title):
    fig,ax=_base_ax()
    conns=[('X','HA1',f"w={W1[0]:.3f}",+1),('X','HA2',f"w={W1[1]:.3f}",-1),
           ('HA1','HB1',f"w={W2[0,0]:.3f}",+1),('HA1','HB2',f"w={W2[0,1]:.3f}",-1),
           ('HA2','HB1',f"w={W2[1,0]:.3f}",+1),('HA2','HB2',f"w={W2[1,1]:.3f}",-1),
           ('HB1','Y',f"w={W3[0]:.3f}",+1),('HB2','Y',f"w={W3[1]:.3f}",-1)]
    for s,d,lb,sg in conns: _arrow(ax,s,d,TEAL,lb,sg)
    _nodes(ax,{'X':X,'HA1':hA[0],'HA2':hA[1],'HB1':hB[0],'HB2':hB[1],'Y':yp})
    ax.set_title(title,fontsize=10.5,color=TEXT,pad=8)
    plt.tight_layout(); return _b64(fig)

def plot_backprop(X,dW1,dW2,dW3,dhA,dhB,dY,title,grad_mags=None):
    fig,ax=_base_ax()
    conns=[('Y','HB1',f"∇={dW3[0]:.4f}",+1),('Y','HB2',f"∇={dW3[1]:.4f}",-1),
           ('HB1','HA1',f"∇={dW2[0,0]:.4f}",+1),('HB2','HA1',f"∇={dW2[0,1]:.4f}",-1),
           ('HB1','HA2',f"∇={dW2[1,0]:.4f}",+1),('HB2','HA2',f"∇={dW2[1,1]:.4f}",-1),
           ('HA1','X',f"∇={dW1[0]:.4f}",+1),('HA2','X',f"∇={dW1[1]:.4f}",-1)]
    for s,d,lb,sg in conns: _arrow(ax,s,d,ORANGE,lb,sg)
    _nodes(ax,{'X':X,'HA1':dhA[0],'HA2':dhA[1],'HB1':dhB[0],'HB2':dhB[1],'Y':dY},
           delta=True,grad_mags=grad_mags)
    ax.set_title(title,fontsize=10.5,color=TEXT,pad=8)
    plt.tight_layout(); return _b64(fig)

# ── Novos gráficos pedagógicos ─────────────────────────────────────────────────
def plot_gradient_flow(dhA,dhB,dY,act_name):
    mag_Y  =float(abs(dY))
    mag_hB =float(np.mean(np.abs(dhB)))
    mag_hA =float(np.mean(np.abs(dhA)))
    vals=[mag_hA,mag_hB,mag_Y]
    layers=['Oculta A\n(mais distante)','Oculta B','Saída']
    max_v=max(vals) if max(vals)>0 else 1.0
    colors=[mcolors.to_hex(mcm.RdYlGn(v/max_v)) for v in vals]

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    y_pos=np.arange(3)

    # Magnitudes absolutas
    bars=axes[0].barh(y_pos,vals,color=colors,edgecolor='#1e2d45',height=0.5)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(layers,fontsize=9)
    axes[0].set_xlabel('Magnitude |δ| média')
    axes[0].set_title(f'Fluxo do Gradiente por Camada  [{act_name}]',color=TEXT,fontsize=11)
    axes[0].set_xlim(0,max_v*1.5)
    for bar,v in zip(bars,vals):
        axes[0].text(bar.get_width()+max_v*0.02,bar.get_y()+bar.get_height()/2,
                    f'{v:.6f}',va='center',fontsize=8.5,color=TEXT)

    # Razão: fração do gradiente da saída que chega a cada camada
    ratios=[mag_hA/mag_Y if mag_Y>0 else 0,
            mag_hB/mag_Y if mag_Y>0 else 0,
            1.0]
    rcolors=[mcolors.to_hex(mcm.RdYlGn(min(r,1))) for r in ratios]
    bars2=axes[1].barh(y_pos,ratios,color=rcolors,edgecolor='#1e2d45',height=0.5)
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels(layers,fontsize=9)
    axes[1].set_xlabel('Fração do gradiente da saída  (1.0 = sem atenuação)')
    axes[1].set_title('Atenuação camada a camada',color=TEXT,fontsize=11)
    axes[1].axvline(1.0,color=MUTED,ls='--',lw=1)
    axes[1].set_xlim(0,max(1.5,max(ratios)*1.35))
    for bar,r in zip(bars2,ratios):
        axes[1].text(bar.get_width()+0.02,bar.get_y()+bar.get_height()/2,
                    f'{r:.3f}×',va='center',fontsize=8.5,color=TEXT)

    plt.tight_layout(); return _b64(fig)

def plot_loss_landscape(X,Y_target,W1_fixed,W2_fixed,hist_w3,act_fn,act_name):
    W3_init=hist_w3[0]
    all_w0=[w[0] for w in hist_w3]; all_w1=[w[1] for w in hist_w3]
    margin=1.2
    r0=max(float(max(abs(np.array(all_w0)-W3_init[0])))+margin,1.0)
    r1=max(float(max(abs(np.array(all_w1)-W3_init[1])))+margin,1.0)
    grid=42
    w0v=np.linspace(W3_init[0]-r0,W3_init[0]+r0,grid)
    w1v=np.linspace(W3_init[1]-r1,W3_init[1]+r1,grid)
    Z=np.zeros((grid,grid))

    zA_f=X*W1_fixed; hA_f=act_fn(zA_f)
    zB_f=hA_f@W2_fixed; hB_f=act_fn(zB_f)

    for i,w0 in enumerate(w0v):
        for j,w1 in enumerate(w1v):
            zY=hB_f[0]*w0+hB_f[1]*w1
            yp=sigmoid(zY)
            Z[j,i]=0.5*(Y_target-yp)**2

    fig,ax=plt.subplots(figsize=(9,7))
    cf=ax.contourf(w0v,w1v,Z,levels=25,cmap='RdYlGn_r',alpha=0.9)
    ax.contour(w0v,w1v,Z,levels=12,colors='white',alpha=0.2,linewidths=0.5)
    plt.colorbar(cf,ax=ax,label='Erro MSE',shrink=0.8)

    traj=np.array(hist_w3)
    if len(traj)>1:
        ax.plot(traj[:,0],traj[:,1],'w--',lw=1.5,alpha=0.7,zorder=6)
        n_arrows=min(6,len(traj)-1)
        step=max(1,(len(traj)-1)//n_arrows)
        for k in range(0,len(traj)-1,step):
            dx=traj[k+1,0]-traj[k,0]; dy_=traj[k+1,1]-traj[k,1]
            if np.sqrt(dx**2+dy_**2)>1e-8:
                ax.annotate("",xy=(traj[k+1,0],traj[k+1,1]),xytext=(traj[k,0],traj[k,1]),
                            arrowprops=dict(arrowstyle='->',color='white',lw=1.5))

    ax.scatter([W3_init[0]],[W3_init[1]],c='yellow',s=180,zorder=8,
               marker='o',edgecolors='black',lw=0.5,label='Início')
    ax.scatter([traj[-1,0]],[traj[-1,1]],c=TEAL,s=220,zorder=8,
               marker='*',edgecolors='black',lw=0.5,label='Final')

    ax.set_xlabel('W3[0]',color=TEXT,fontsize=11)
    ax.set_ylabel('W3[1]',color=TEXT,fontsize=11)
    ax.set_title(f'Loss Landscape — erro em função de W3  [{act_name}]\n'
                 '(W1 e W2 fixos nos valores iniciais; linha = trajetória do treinamento)',
                 color=TEXT,fontsize=10)
    ax.tick_params(colors=MUTED); ax.legend(fontsize=9)
    plt.tight_layout(); return _b64(fig)

def plot_weight_delta(W1b,W1,W2b,W2,W3b,W3):
    labels=['W1[0]','W1[1]','W2[0,0]','W2[0,1]','W2[1,0]','W2[1,1]','W3[0]','W3[1]']
    before=[W1b[0],W1b[1],W2b[0,0],W2b[0,1],W2b[1,0],W2b[1,1],W3b[0],W3b[1]]
    after =[W1[0], W1[1], W2[0,0], W2[0,1], W2[1,0], W2[1,1], W3[0], W3[1]]
    deltas=[a-b for a,b in zip(after,before)]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    y=np.arange(len(labels))

    axes[0].barh(y-0.22,before,height=0.38,color=MUTED,alpha=0.75,label='Antes')
    axes[0].barh(y+0.22,after, height=0.38,color=TEAL, alpha=0.9, label='Depois')
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels,fontsize=8.5)
    axes[0].axvline(0,color='white',alpha=0.3,lw=0.8)
    axes[0].set_title('Valores Antes vs. Depois da Época 1',color=TEXT,fontsize=11)
    axes[0].legend(fontsize=9)

    dcols=[TEAL if d>0 else ORANGE for d in deltas]
    axes[1].barh(y,deltas,height=0.5,color=dcols,edgecolor='none')
    axes[1].set_yticks(y); axes[1].set_yticklabels(labels,fontsize=8.5)
    axes[1].axvline(0,color='white',alpha=0.6,lw=1.5)
    axes[1].set_title('Δ por peso  (teal=cresceu, laranja=diminuiu)',color=TEXT,fontsize=10)
    max_d=max(abs(d) for d in deltas) if any(deltas) else 1.0
    for i,d in enumerate(deltas):
        offset=max_d*0.04; ha='left' if d>=0 else 'right'
        axes[1].text(d+(offset if d>=0 else -offset),i,f'{d:+.5f}',
                    va='center',ha=ha,fontsize=7.5,color=TEXT)
    plt.tight_layout(); return _b64(fig)

def plot_learning_curve(hist_err,hist_pred,Y,lr,X,epochs,act_name):
    ep=list(range(1,len(hist_err)+1))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(ep,hist_err,color=ORANGE,lw=2)
    marks=sorted(set([0,min(9,len(hist_err)-1),len(hist_err)//2,len(hist_err)-1]))
    axes[0].scatter([ep[m] for m in marks],[hist_err[m] for m in marks],
                    color='#c0392b',zorder=5,s=55)
    axes[0].set_title('Curva de Aprendizado — Erro MSE',fontsize=11,color=TEXT)
    axes[0].set_xlabel('Épocas'); axes[0].set_ylabel('Erro (MSE)')
    axes[1].plot(ep,hist_pred,color=TEAL,lw=2,label='Predição ŷ')
    axes[1].axhline(Y,color='#27ae60',ls='--',lw=2,label=f'Alvo y = {Y}')
    axes[1].set_title('Predição convergindo para o alvo',fontsize=11,color=TEXT)
    axes[1].set_xlabel('Épocas'); axes[1].set_ylabel('ŷ'); axes[1].legend()
    plt.suptitle(f'X={X}, alvo={Y}, lr={lr}, {epochs} épocas  [{act_name}]',
                 color=TEXT,fontsize=11)
    plt.tight_layout(); return _b64(fig)


# ── Textos descritivos por função de ativação ──────────────────────────────────
ACT_INTRO={
'sigmoid':(
    "A sigmoide foi a função padrão das primeiras redes neurais. "
    "Ela comprime qualquer entrada para o intervalo (0,1), o que a torna útil na camada de saída "
    "para problemas de classificação binária. Nas camadas ocultas, porém, ela tem um problema sério.",
    r"\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \in (0,\,1)",
    r"\sigma'(z) = \sigma(z)\cdot(1-\sigma(z)) \;\leq\; 0.25",
    "O gradiente máximo da sigmoide é apenas 0.25 (em z=0). "
    "Em redes profundas, esse gradiente é multiplicado camada a camada: "
    "0.25³ = 0.016 (3 camadas) → o sinal de erro quase desaparece antes de chegar à primeira camada. "
    "Isso é o vanishing gradient — o principal motivo pelo qual redes profundas com sigmoide não treinam bem.",
    "Atenção: para |z| > 3 o gradiente é praticamente zero. "
    "Neurônios saturados param de aprender (dying sigmoid).",
    'orange'
),
'relu':(
    "A ReLU (Rectified Linear Unit) domina redes profundas modernas por uma razão simples: "
    "ela resolve o vanishing gradient. Para z > 0, o gradiente é exatamente 1 — "
    "não importa quantas camadas profundas o sinal percorra.",
    r"f(z) = \max(0,\, z) = \begin{cases} 0 & z \leq 0 \\ z & z > 0 \end{cases}",
    r"f'(z) = \begin{cases} 0 & z \leq 0 \\ 1 & z > 0 \end{cases}",
    "Zona morta (dying ReLU): se z ≤ 0, o gradiente é exatamente zero. "
    "Com learning rate alto, pesos podem ser empurrados para valores que mantêm todos os neurônios "
    "negativos permanentemente — eles 'morrem' e param de aprender. "
    "Leaky ReLU e ELU foram criadas para resolver esse problema.",
    "Vantagem chave: gradiente constante (=1) na zona ativa elimina o vanishing gradient.",
    'teal'
),
'tanh':(
    "A Tanh (tangente hiperbólica) é uma versão centrada em zero da sigmoide. "
    "Enquanto σ(z) ∈ (0,1), tanh(z) ∈ (-1,1). Isso é importante: "
    "saídas centradas em zero reduzem o zig-zag na descida do gradiente e ajudam na convergência.",
    r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \qquad \in (-1,\,1)",
    r"\tanh'(z) = 1 - \tanh(z)^2 \;\leq\; 1",
    "Derivada máxima = 1 (em z=0) — quatro vezes maior que a da sigmoide. "
    "Isso reduz (mas não elimina) o vanishing gradient. Para |z| grande, tanh' ainda vai a zero. "
    "Históricamente, foi a função padrão em RNNs e LSTMs antes da popularização da ReLU.",
    "Melhor que a sigmoide para camadas ocultas (zero-centrada, gradiente maior), "
    "mas ainda sofre de vanishing gradient para z grande.",
    'teal'
),
'leaky_relu':(
    "A Leaky ReLU foi criada para resolver o dying ReLU. "
    "A modificação é mínima: o lado negativo, em vez de zero, recebe uma pequena inclinação α. "
    "Com α = 0.01, o gradiente nunca é exatamente zero — "
    "cada neurônio sempre contribui com pelo menos um sinal mínimo.",
    r"f(z) = \begin{cases} \alpha\, z & z \leq 0 \\ z & z > 0 \end{cases} \qquad \alpha = 0.01",
    r"f'(z) = \begin{cases} \alpha & z \leq 0 \\ 1 & z > 0 \end{cases}",
    "A inclinação negativa α é um hiperparâmetro. Valores comuns: 0.01 a 0.3. "
    "Com α muito alto a rede deixa de ter a 'retificação' que dá à ReLU sua vantagem. "
    "Uma variante chamada PReLU torna α aprendível (treinado junto com os outros pesos).",
    "Gradiente nunca zero: cada neurônio sempre aprende, mesmo com z < 0. "
    "Resolução direta do dying ReLU com custo computacional mínimo.",
    'teal'
),
'elu':(
    "A ELU (Exponential Linear Unit) leva a ideia da Leaky ReLU mais longe: "
    "em vez de uma reta negativa, usa uma curva exponencial suave. "
    "Isso faz a média das ativações ficar mais próxima de zero "
    "(como a Tanh), reduzindo o deslocamento do viés e acelerando a convergência.",
    r"f(z) = \begin{cases} \alpha(e^z - 1) & z \leq 0 \\ z & z > 0 \end{cases} \qquad \alpha = 1.0",
    r"f'(z) = \begin{cases} \alpha\, e^z & z \leq 0 \\ 1 & z > 0 \end{cases}",
    "Para z muito negativo, a ELU satura em −α = −1. "
    "Isso difere da Leaky ReLU (que cresce linearmente para z → −∞) e reduz o ruído em z negativo. "
    "A derivada é suave em z=0 (exp(0)=1), ao contrário da ReLU que tem um ponto anguloso.",
    "Smooth em todos os pontos + zero-centrada + sem dying ReLU. "
    "Custo: a exponencial é mais cara computacionalmente que max(0,z).",
    'teal'
),
'swish':(
    "A Swish (também chamada SiLU) foi descoberta por busca neural automática (AutoML). "
    "É definida como z·σ(z) — o próprio z modulado por uma porta sigmoid. "
    "Pense assim: para z positivo grande, σ(z)≈1 então Swish≈z (como ReLU). "
    "Para z negativo, σ(z)≈0 e Swish≈0. Mas na transição, ela é completamente suave.",
    r"f(z) = z\cdot\sigma(z) = \frac{z}{1+e^{-z}}",
    r"f'(z) = \sigma(z) + z\cdot\sigma(z)\cdot(1-\sigma(z)) = \sigma(z)(1 + z(1-\sigma(z)))",
    "A Swish é levemente não-monótona: tem um mínimo em z ≈ -1.3 (valor ≈ -0.28). "
    "Isso permite que a rede 'se lembre' de valores negativos moderados, o que pesquisas "
    "mostraram ser útil em algumas tarefas. É usada em EfficientNet, MobileNetV3 e variantes do GPT.",
    "Suave em todos os pontos, sem dying ReLU, gradiente rico na transição. "
    "Custo: precisa calcular σ(z) além de z, ligeiramente mais lento que ReLU.",
    'teal'
),
}


# ── Função principal ───────────────────────────────────────────────────────────
def nn_run_all(X_s,Y_s,lr_s,w1_0,w1_1,w2_00,w2_01,w2_10,w2_11,w3_0,w3_1,epochs_s,act_s="sigmoid"):
    try:
        X=float(X_s); Y=float(Y_s); lr=float(lr_s); epochs=int(epochs_s)
        W1=np.array([float(w1_0),float(w1_1)])
        W2=np.array([[float(w2_00),float(w2_01)],[float(w2_10),float(w2_11)]])
        W3=np.array([float(w3_0),float(w3_1)])

        act_key=act_s if act_s in ACT_FNS else 'sigmoid'
        act_fn,dact_z,act_name,_=ACT_FNS[act_key]

        # Guarda W iniciais para o loss landscape e delta final
        W1_init=W1.copy(); W2_init=W2.copy(); W3_init_full=W3.copy()
        hist_w3=[W3.copy()]  # trajetória do W3 (para landscape)

        steps=[]

        # ── Passo 0: Configuração ────────────────────────────────────────────
        steps.append({"title":"Configuração da Rede","sections":[
            {"type":"text","content":
             f"Vamos construir uma MLP com 4 camadas: Entrada (1), Oculta A (2), Oculta B (2), Saída (1). "
             f"As camadas ocultas usarão {act_name}; a saída usa sempre Sigmoide para manter ŷ ∈ (0,1)."},
            {"type":"math","content":
             r"X \xrightarrow{W_1} h_A \xrightarrow{W_2} h_B \xrightarrow{W_3} \hat{y}"},
            {"type":"subtitle","content":"Por que os pesos iniciais importam?"},
            {"type":"text","content":
             "Os pesos são os parâmetros que a rede aprende. Antes do treinamento precisamos de valores de partida. "
             "Não podemos começar em zero: se W=0 todos os neurônios produzem o mesmo resultado e o gradiente também é zero — "
             "a rede nunca sairia do lugar (problema da simetria). "
             "Valores diferentes quebram essa simetria e permitem especialização."},
            {"type":"subtitle","content":"O que cada matriz conecta"},
            {"type":"table","headers":["Matriz","Dimensão","Conecta","Interpretação"],"rows":[
                ["W1","1 × 2","X → Oculta    A",
                 f"W1[0]={w1_0} escala X para hA1;  W1[1]={w1_1} escala X para hA2"],
                ["W2","2 × 2","Oculta A → Oculta B",
                 "Cada coluna combina hA1 e hA2 para produzir um neurônio de hB"],
                ["W3","2 × 1","Oculta B → Saída",
                 f"W3[0]={w3_0} pondera hB1;  W3[1]={w3_1} pondera hB2"],
            ]},
            {"type":"table","headers":["Parâmetro","Valor","Papel"],"rows":[
                ["X",str(X),"Entrada da rede"],
                ["y",str(Y),"Alvo que queremos aprender"],
                ["lr",str(lr),"Tamanho do passo de ajuste"],
                ["Épocas",str(epochs),"Repetições do ciclo forward→backprop→update"],
                ["Ativação",act_name,"Função usada nas camadas ocultas"],
            ]},
            {"type":"highlight","content":f"Objetivo: dado X={X}, ajustar W1,W2,W3 até ŷ ≈ {Y}.","variant":"teal"},
        ]})

        # ── Passo 1: Função de Ativação ───────────────────────────────────────
        intro,form_fn,form_deriv,body,note,note_var=ACT_INTRO[act_key]
        z_sample=[-4,-2,-1,0,1,2,4]
        trows=[]
        for zv in z_sample:
            za=np.array([float(zv)]); fv=float(act_fn(za)[0]); dv=float(dact_z(za)[0])
            trows.append([str(zv),f'{fv:.4f}',f'{dv:.4f}'])

        act_sec=[
            {"type":"text","content":intro},
            {"type":"math","content":form_fn},
            {"type":"text","content":
             "A derivada é usada no backpropagation para saber em quanto o erro muda ao variar z:"},
            {"type":"math","content":form_deriv},
            {"type":"img","content":plot_activation(act_key)},
            {"type":"subtitle","content":"Valores numéricos em pontos chave"},
            {"type":"table","headers":["z","f(z)","f'(z) — gradiente"],"rows":trows},
            {"type":"text","content":body},
            {"type":"highlight","content":note,"variant":note_var},
            {"type":"subtitle","content":"Comparativo — todas as 6 funções disponíveis"},
            {"type":"text","content":
             "A escolha da função de ativação afeta diretamente como o gradiente flui pelas camadas. "
             "O gráfico abaixo mostra saída e derivada de todas as funções na mesma escala:"},
            {"type":"img","content":plot_activation_comparison()},
        ]
        steps.append({"title":f"Função de Ativação: {act_name}","sections":act_sec})

        # ── Forward Pass Época 1 ─────────────────────────────────────────────
        zA=X*W1; hA=act_fn(zA)
        zB=hA@W2; hB=act_fn(zB)
        zY=hB@W3; yp=sigmoid(zY)
        err1=0.5*(Y-yp)**2
        act_sym=act_name

        steps.append({"title":"Forward Pass — Época 1","sections":[
            {"type":"text","content":
             f"Cada camada multiplica os valores anteriores pelos pesos e aplica {act_sym}. "
             "Abaixo cada cálculo com os números reais."},
            {"type":"img","content":plot_forward(X,hA,hB,yp,W1,W2,W3,
                f"Época 1 — Forward Pass  (setas: pesos;  nós: ativações {act_sym}/σ)")},
            {"type":"subtitle","content":f"① Oculta A  —  z_A = X·W1,  h_A = {act_sym}(z_A)"},
            {"type":"math","content":
             r"z_{A_1}=X\times W_1[0]="+f"{X}\\times{W1[0]}={zA[0]:.4f}"},
            {"type":"math","content":
             r"z_{A_2}=X\times W_1[1]="+f"{X}\\times{W1[1]}={zA[1]:.4f}"},
            {"type":"math","content":
             f"h_A={act_sym}(z_A)=[{act_sym}({zA[0]:.4f}),\\ {act_sym}({zA[1]:.4f})]=[{hA[0]:.4f},\\ {hA[1]:.4f}]"},
            {"type":"subtitle","content":f"② Oculta B  —  z_B = h_A·W2,  h_B = {act_sym}(z_B)"},
            {"type":"math","content":
             r"z_{B_1}=h_{A_1}\times W_2[0,0]+h_{A_2}\times W_2[1,0]="
             +f"{hA[0]:.4f}\\times{W2[0,0]}+{hA[1]:.4f}\\times{W2[1,0]}={zB[0]:.4f}"},
            {"type":"math","content":
             r"z_{B_2}=h_{A_1}\times W_2[0,1]+h_{A_2}\times W_2[1,1]="
             +f"{hA[0]:.4f}\\times{W2[0,1]}+{hA[1]:.4f}\\times{W2[1,1]}={zB[1]:.4f}"},
            {"type":"math","content":
             f"h_B={act_sym}(z_B)=[{hB[0]:.4f},\\ {hB[1]:.4f}]"},
            {"type":"subtitle","content":"③ Saída  —  z_Y = h_B·W3,  ŷ = σ(z_Y)"},
            {"type":"math","content":
             r"z_Y=h_{B_1}\times W_3[0]+h_{B_2}\times W_3[1]="
             +f"{hB[0]:.4f}\\times{W3[0]}+{hB[1]:.4f}\\times{W3[1]}={zY:.4f}"},
            {"type":"math","content":
             r"\hat{y}=\sigma(z_Y)=\frac{1}{1+e^{-("+f"{zY:.4f}"
             +r")}}="+f"{yp:.4f}"},
            {"type":"highlight","content":
             f"Predição: ŷ = {yp:.4f}  |  Alvo: y = {Y}  |  Erro: E = {err1:.6f}","variant":"orange"},
        ]})

        # ── Função de Custo ───────────────────────────────────────────────────
        steps.append({"title":"Função de Custo — Época 1","sections":[
            {"type":"text","content":
             "Precisamos de um número que quantifica o quão errada está a rede. "
             "O MSE penaliza erros grandes mais que erros pequenos, "
             "e o fator ½ cancela o 2 que aparece na derivada."},
            {"type":"math","content":r"E=\frac{1}{2}(y-\hat{y})^2"},
            {"type":"math","content":
             r"E=\frac{1}{2}("+f"{Y}-{yp:.4f}"
             +r")^2=\frac{1}{2}\times("+f"{Y-yp:.4f}"
             +r")^2=\frac{1}{2}\times"+f"{(Y-yp)**2:.6f}"
             +r"="+f"{err1:.6f}"},
            {"type":"highlight","content":
             f"A rede prevê {yp:.4f}, deveria ser {Y}. "
             "O backpropagation calculará como mover cada peso para reduzir esse valor.",
             "variant":"orange"},
        ]})

        # ── Loss Landscape (calculado depois do treinamento, inserido aqui) ──
        # placeholder — será preenchido após o treinamento completo
        landscape_placeholder_idx=len(steps)
        steps.append(None)  # reserva posição

        # ── Backpropagation Época 1 ───────────────────────────────────────────
        W1b=W1.copy(); W2b=W2.copy(); W3b=W3.copy()
        dsY=d_sig(yp); dY=(yp-Y)*dsY; dW3=dY*hB
        dsB=dact_z(zB); dhB=dY*W3*dsB; dW2=np.outer(hA,dhB)
        dsA=dact_z(zA); dhA=dhB@W2.T*dsA; dW1=dhA*X
        W3-=lr*dW3; W2-=lr*dW2; W1-=lr*dW1
        hist_w3.append(W3.copy())

        grad_mags_ep1={
            'Y':float(abs(dY)),
            'HB1':float(abs(dhB[0])),'HB2':float(abs(dhB[1])),
            'HA1':float(abs(dhA[0])),'HA2':float(abs(dhA[1])),
        }
        ratio_hA=np.mean(np.abs(dhA))/max(abs(dY),1e-12)
        vanishing_note=""
        if ratio_hA<0.1:
            vanishing_note=(f"⚠️ O gradiente médio em Oculta A é {ratio_hA*100:.1f}% do gradiente da saída — "
                           f"sinal claro de vanishing gradient com {act_name}.")
        elif ratio_hA>0.9:
            vanishing_note=(f"✔ Gradiente bem preservado em Oculta A ({ratio_hA*100:.1f}% do da saída) — "
                           f"{act_name} mantem fluxo saudável.")

        form_d_act = (
            r"f'(z)=\begin{cases}0&z\leq0\\1&z>0\end{cases}"
            if act_key in ('relu','leaky_relu') else
            r"f'(z)=1-\tanh(z)^2" if act_key=='tanh' else
            r"f'(z)=\sigma(z)(1-\sigma(z))" if act_key=='sigmoid' else
            r"f'(z)=\alpha e^z\,(z\leq0),\;1\,(z>0)" if act_key=='elu' else
            r"f'(z)=\sigma(z)(1+z(1-\sigma(z)))"
        )
        bp_sections=[
            {"type":"text","content":
             "O backpropagation aplica a regra da cadeia de trás para frente. "
             "Cada δ mede a 'culpa' de um neurônio no erro final. "
             "Os nós agora são coloridos: verde = gradiente forte, amarelo = moderado, vermelho = fraco."},
            {"type":"img","content":plot_backprop(X,dW1,dW2,dW3,dhA,dhB,dY,
                f"Época 1 — Backpropagation  (nós coloridos por magnitude de δ)",
                grad_mags=grad_mags_ep1)},
            {"type":"subtitle","content":"① δ_Y — delta da saída (sigmoide)"},
            {"type":"math","content":
             r"\sigma'(\hat{y})=\hat{y}(1-\hat{y})="+f"{yp:.4f}\\times{1-yp:.4f}={dsY:.4f}"},
            {"type":"math","content":
             r"\delta_Y=(\hat{y}-y)\cdot\sigma'(\hat{y})="
             +f"({yp:.4f}-{Y})\\times{dsY:.4f}={dY:.6f}"},
            {"type":"subtitle","content":f"② ∇W3  e  δ_hB — derivada de {act_name}"},
            {"type":"math","content":
             form_d_act+r"\;\Rightarrow\; f'(z_{B_1})="+f"{dsB[0]:.4f}"
             +r",\; f'(z_{B_2})="+f"{dsB[1]:.4f}"},
            {"type":"math","content":
             r"\nabla W_3[0]=\delta_Y\times h_{B_1}="
             +f"{dY:.4f}\\times{hB[0]:.4f}={dW3[0]:.4f}"},
            {"type":"math","content":
             r"\delta_{h_{B_1}}=\delta_Y\times W_3[0]\times f'(z_{B_1})="
             +f"{dY:.4f}\\times{W3b[0]:.4f}\\times{dsB[0]:.4f}={dhB[0]:.6f}"},
            {"type":"subtitle","content":"③ ∇W2"},
            {"type":"math","content":
             r"\nabla W_2=\begin{bmatrix}"
             +f"{hA[0]:.3f}\\times{dhB[0]:.4f}&{hA[0]:.3f}\\times{dhB[1]:.4f}"
             +r"\\"+f"{hA[1]:.3f}\\times{dhB[0]:.4f}&{hA[1]:.3f}\\times{dhB[1]:.4f}"
             +r"\end{bmatrix}=\begin{bmatrix}"
             +f"{dW2[0,0]:.4f}&{dW2[0,1]:.4f}"
             +r"\\"+f"{dW2[1,0]:.4f}&{dW2[1,1]:.4f}"
             +r"\end{bmatrix}"},
            {"type":"subtitle","content":"④ δ_hA e ∇W1"},
            {"type":"math","content":
             r"f'(z_{A_1})="+f"{dsA[0]:.4f}"+r",\quad f'(z_{A_2})="+f"{dsA[1]:.4f}"},
            {"type":"math","content":
             r"\delta_{h_{A_1}}=(\delta_{h_{B_1}}\cdot W_2[0,0]+\delta_{h_{B_2}}\cdot W_2[0,1])\cdot f'(z_{A_1})="
             +f"({dhB[0]:.4f}\\cdot{W2b[0,0]}+{dhB[1]:.4f}\\cdot{W2b[0,1]})\\cdot{dsA[0]:.4f}={dhA[0]:.6f}"},
            {"type":"math","content":
             r"\nabla W_1[0]=\delta_{h_{A_1}}\times X="
             +f"{dhA[0]:.6f}\\times{X}={dW1[0]:.6f}"},
            {"type":"subtitle","content":f"⑤ Atualização  (W ← W − {lr} · ∇W)"},
            {"type":"table","headers":["Peso","Antes","−lr·∇","Depois"],"rows":[
                ["W3[0]",f"{W3b[0]:.4f}",f"−{lr}×{dW3[0]:.4f}={-lr*dW3[0]:.4f}",f"{W3[0]:.4f}"],
                ["W3[1]",f"{W3b[1]:.4f}",f"−{lr}×{dW3[1]:.4f}={-lr*dW3[1]:.4f}",f"{W3[1]:.4f}"],
                ["W2[0,0]",f"{W2b[0,0]:.4f}",f"−{lr}×{dW2[0,0]:.4f}={-lr*dW2[0,0]:.4f}",f"{W2[0,0]:.4f}"],
                ["W2[0,1]",f"{W2b[0,1]:.4f}",f"−{lr}×{dW2[0,1]:.4f}={-lr*dW2[0,1]:.4f}",f"{W2[0,1]:.4f}"],
                ["W2[1,0]",f"{W2b[1,0]:.4f}",f"−{lr}×{dW2[1,0]:.4f}={-lr*dW2[1,0]:.4f}",f"{W2[1,0]:.4f}"],
                ["W2[1,1]",f"{W2b[1,1]:.4f}",f"−{lr}×{dW2[1,1]:.4f}={-lr*dW2[1,1]:.4f}",f"{W2[1,1]:.4f}"],
                ["W1[0]",f"{W1b[0]:.4f}",f"−{lr}×{dW1[0]:.4f}={-lr*dW1[0]:.4f}",f"{W1[0]:.4f}"],
                ["W1[1]",f"{W1b[1]:.4f}",f"−{lr}×{dW1[1]:.4f}={-lr*dW1[1]:.4f}",f"{W1[1]:.4f}"],
            ]},
            {"type":"subtitle","content":"⑥ Fluxo do gradiente por camada"},
            {"type":"text","content":
             "Quanto do sinal de erro chegou a cada camada? "
             "O gráfico mostra a magnitude média de δ e a fração preservada em relação à saída."},
            {"type":"img","content":plot_gradient_flow(dhA,dhB,dY,act_name)},
            {"type":"subtitle","content":"⑦ Pesos antes vs. depois (Δ)"},
            {"type":"img","content":plot_weight_delta(W1b,W1,W2b,W2,W3b,W3)},
        ]
        if vanishing_note:
            bp_sections.append({"type":"highlight","content":vanishing_note,
                                 "variant":"orange" if ratio_hA<0.1 else "teal"})
        steps.append({"title":"Backpropagation — Época 1","sections":bp_sections})

        # ── Época 2 ───────────────────────────────────────────────────────────
        zA2=X*W1; hA2=act_fn(zA2); zB2=hA2@W2; hB2=act_fn(zB2)
        zY2=hB2@W3; yp2=sigmoid(zY2); err2=0.5*(Y-yp2)**2
        dY2=(yp2-Y)*d_sig(yp2)
        dW3_2=dY2*hB2; dhB2=dY2*W3*dact_z(zB2)
        dW2_2=np.outer(hA2,dhB2); dhA2=dhB2@W2.T*dact_z(zA2); dW1_2=dhA2*X
        W3-=lr*dW3_2; W2-=lr*dW2_2; W1-=lr*dW1_2
        hist_w3.append(W3.copy())

        steps.append({"title":"Época 2 — Ciclo Completo","sections":[
            {"type":"text","content":
             "Repetimos o mesmo ciclo com pesos já modificados. O erro deve diminuir."},
            {"type":"img","content":plot_forward(X,hA2,hB2,yp2,W1,W2,W3,
                "Época 2 — Forward Pass (pesos ajustados pela época 1)")},
            {"type":"math","content":
             r"h_A=["+f"{hA2[0]:.4f},\\ {hA2[1]:.4f}"
             +r"]\qquad h_B=["+f"{hB2[0]:.4f},\\ {hB2[1]:.4f}"+r"]"},
            {"type":"math","content":
             r"\hat{y}="+f"{yp2:.4f}"+r"\qquad E="+f"{err2:.6f}"
             +r"\quad(\text{época 1: }"+f"{err1:.6f}"+r")"},
            {"type":"highlight","content":
             f"Redução:  {err1:.6f} → {err2:.6f}  ({(1-err2/err1)*100:.1f}% menor)",
             "variant":"teal"},
            {"type":"img","content":plot_backprop(X,dW1_2,dW2_2,dW3_2,dhA2,dhB2,dY2,
                "Época 2 — Backpropagation")},
            {"type":"math","content":
             r"\delta_Y="+f"{dY2:.6f}"+r"\quad(\text{época 1: }"+f"{dY:.6f}"+r")"},
            {"type":"text","content":
             "Os gradientes diminuíram — sinal de que estamos mais perto do mínimo."},
        ]})

        # ── Treinamento completo ───────────────────────────────────────────────
        hist_e=[err1,err2]; hist_p=[yp,yp2]
        sample_iv=max(1,epochs//12)
        for ep_i in range(3,epochs+1):
            zA_=X*W1; hA_=act_fn(zA_); zB_=hA_@W2; hB_=act_fn(zB_)
            zY_=hB_@W3; yp_=sigmoid(zY_); e_=0.5*(Y-yp_)**2
            hist_e.append(e_); hist_p.append(yp_)
            dY_=(yp_-Y)*d_sig(yp_); dW3_=dY_*hB_
            dhB_=dY_*W3*dact_z(zB_); dW2_=np.outer(hA_,dhB_)
            dhA_=dhB_@W2.T*dact_z(zA_); dW1_=dhA_*X
            W3-=lr*dW3_; W2-=lr*dW2_; W1-=lr*dW1_
            if ep_i%sample_iv==0 or ep_i==epochs:
                hist_w3.append(W3.copy())

        marks=sorted(set([max(0,x) for x in [0,9,epochs//4-1,epochs//2-1,epochs-1]]))
        rows=[[str(m+1),f"{hist_p[m]:.4f}",f"{hist_e[m]:.6f}"] for m in marks]

        steps.append({"title":f"Treinamento Completo — {epochs} Épocas","sections":[
            {"type":"text","content":
             f"Épocas 1 e 2 foram detalhadas nos passos anteriores. "
             f"Aqui executamos o ciclo para as épocas 3 a {epochs}."},
            {"type":"img","content":plot_learning_curve(hist_e,hist_p,Y,lr,X,epochs,act_name)},
            {"type":"table","headers":["Época","Predição ŷ","Erro (MSE)"],"rows":rows},
            {"type":"highlight","content":
             f"Redução total: {hist_e[0]:.6f} → {hist_e[-1]:.6f}  "
             f"({(1-hist_e[-1]/hist_e[0])*100:.1f}% de melhora em {epochs} épocas)","variant":"teal"},
        ]})

        # ── Estado Final ──────────────────────────────────────────────────────
        zA_f=X*W1; hA_f=act_fn(zA_f); zB_f=hA_f@W2; hB_f=act_fn(zB_f)
        zY_f=hB_f@W3; yp_f=sigmoid(zY_f)
        steps.append({"title":"Estado Final da Rede","sections":[
            {"type":"img","content":plot_forward(X,hA_f,hB_f,yp_f,W1,W2,W3,
                f"Após {epochs} épocas — ŷ={yp_f:.4f}  |  alvo={Y}")},
            {"type":"math","content":
             r"\hat{y}_{final}="+f"{yp_f:.6f}"
             +r"\quad\longrightarrow\quad y="+f"{Y}"
             +r"\quad(E_{final}="+f"{hist_e[-1]:.8f}"+r")"},
            {"type":"highlight","content":
             f"Evolução:  ŷ={hist_p[0]:.4f} (época 1)  →  {yp_f:.4f} (época {epochs})   alvo={Y}",
             "variant":"teal"},
            {"type":"subtitle","content":"Pesos finais aprendidos"},
            {"type":"table","headers":["Peso","Valor final"],"rows":[
                ["W1[0]",f"{W1[0]:.6f}"],["W1[1]",f"{W1[1]:.6f}"],
                ["W2[0,0]",f"{W2[0,0]:.6f}"],["W2[0,1]",f"{W2[0,1]:.6f}"],
                ["W2[1,0]",f"{W2[1,0]:.6f}"],["W2[1,1]",f"{W2[1,1]:.6f}"],
                ["W3[0]",f"{W3[0]:.6f}"],["W3[1]",f"{W3[1]:.6f}"],
            ]},
        ]})

        # ── Preenche o Loss Landscape (passo 4) ───────────────────────────────
        landscape_img=plot_loss_landscape(X,Y,W1_init,W2_init,hist_w3,act_fn,act_name)
        steps[landscape_placeholder_idx]={
            "title":"Loss Landscape — A Superfície do Erro",
            "sections":[
                {"type":"text","content":
                 "Antes de entender o backpropagation, vale visualizar o problema: "
                 "onde estamos no espaço do erro? "
                 "Este mapa mostra como o MSE varia conforme mudamos apenas W3[0] e W3[1], "
                 "mantendo W1 e W2 fixos nos valores iniciais."},
                {"type":"img","content":landscape_img},
                {"type":"text","content":
                 "Cada ponto no mapa é uma configuração diferente de W3. "
                 "Cores escuras (verde) = erro baixo (vales, mínimos). "
                 "Cores claras (vermelho) = erro alto (picos, planaltos). "
                 "A linha tracejada branca é a trajetória percorrida durante o treinamento — "
                 "o algoritmo navega essa superfície descendo sempre na direção de maior declive (gradiente negativo)."},
                {"type":"highlight","content":
                 "⚠ Este é um corte 2D de um espaço 8-dimensional (8 pesos). "
                 "Na realidade, W1 e W2 também se movem simultaneamente, "
                 "mas este slice já revela o conceito: existem vales, planaltos, e às vezes mínimos locais.",
                 "variant":"orange"},
                {"type":"text","content":
                 "Por que o landscape tem essa forma? A saída ŷ = σ(hB · W3) é uma função sigmoid — "
                 "suave e limitada em (0,1). O erro E = ½(y−ŷ)² é então uma superfície convexa "
                 "em relação a W3 (se W1 e W2 fossem fixos). "
                 "Isso significa que, para W3 isolado, existe um único mínimo global."},
            ]
        }

        # ── Injeta no Estado Final: landscape + variação total dos pesos ─────
        estado_final_secs=steps[-1]['sections']
        estado_final_secs.append({
            "type":"subtitle","content":"Revisitando o Loss Landscape — onde o treinamento chegou"})
        estado_final_secs.append({
            "type":"text","content":
            "O mesmo mapa de erro exibido no passo 4, agora com a trajetória completa. "
            "A estrela teal mostra onde os pesos de W3 estão após todas as épocas — "
            "confirme que o algoritmo de fato desceu em direção ao vale."})
        estado_final_secs.append({"type":"img","content":landscape_img})
        estado_final_secs.append({
            "type":"subtitle",
            "content":f"Variação Total dos Pesos — Início → Final  ({epochs} épocas)"})
        estado_final_secs.append({
            "type":"text","content":
            f"Compara os {8} pesos iniciais (configurados antes da época 1) com os valores "
            f"aprendidos após {epochs} épocas de treinamento. "
            "Barras teal = peso cresceu; barras laranja = peso diminuiu. "
            "A magnitude da barra indica o quanto cada peso se moveu no total."})
        estado_final_secs.append({
            "type":"img","content":
            plot_weight_delta(W1_init,W1,W2_init,W2,W3_init_full,W3)})
        estado_final_secs.append({
            "type":"highlight","variant":"teal","content":
            "Pesos com maior Δ tiveram maior influência no aprendizado. "
            "Pesos que mal se moveram podem indicar neurônios pouco ativados ou gradientes fracos — "
            "o fluxo do gradiente (passo 5) ajuda a diagnosticar a causa."})

        summary={
            "X":X,"Y":Y,"lr":lr,"epochs":epochs,"act":act_name,
            "w1_init":[float(w1_0),float(w1_1)],
            "w2_init":[[float(w2_00),float(w2_01)],[float(w2_10),float(w2_11)]],
            "w3_init":[float(w3_0),float(w3_1)],
            "yp_f":float(yp_f),"err_f":float(hist_e[-1])
        }
        return json.dumps({"ok":True,"steps":steps,"summary":summary})

    except Exception as e:
        import traceback
        return json.dumps({"ok":False,"error":str(e),"tb":traceback.format_exc()})
