import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64, json

plt.rcParams.update({
    'figure.facecolor': '#111827', 'axes.facecolor': '#0a0e1a',
    'axes.edgecolor':   '#1e2d45', 'axes.labelcolor': '#94a3b8',
    'xtick.color':      '#64748b', 'ytick.color':     '#64748b',
    'text.color':       '#e2e8f0', 'grid.color':      '#1e2d45',
    'grid.linewidth':   0.6,       'axes.grid':       True,
    'legend.facecolor': '#111827', 'legend.edgecolor':'#1e2d45',
    'legend.fontsize':  8,
})

TEAL   = '#00d4aa'
ORANGE = '#ff6b35'
MUTED  = '#64748b'
TEXT   = '#e2e8f0'

ACT_COLORS = {
    'sigmoid':    '#00d4aa', 'relu':       '#ff6b35',
    'tanh':       '#6c8ebf', 'leaky_relu': '#ffd700',
    'elu':        '#98fb98', 'swish':      '#da70d6',
}
ACT_NAMES = {
    'sigmoid':'Sigmoide','relu':'ReLU','tanh':'Tanh',
    'leaky_relu':'Leaky ReLU','elu':'ELU','swish':'Swish',
}

# ── Funções de ativação e derivadas ───────────────────────────────────────────
def sigmoid(z):   return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def d_sig_out(a): return a * (1 - a)  # derivada dado a SAÍDA a=σ(z)

def act_fn(z, key):
    if   key == 'sigmoid':    return sigmoid(z)
    elif key == 'relu':       return np.maximum(0.0, z)
    elif key == 'tanh':       return np.tanh(z)
    elif key == 'leaky_relu': return np.where(z > 0, z, 0.01 * z)
    elif key == 'elu':        return np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1)
    elif key == 'swish':      return z * sigmoid(z)
    return sigmoid(z)

def dact(z, key):
    """Derivada dado z (pré-ativação)."""
    if key == 'sigmoid':
        s = sigmoid(z); return s * (1 - s)
    elif key == 'relu':
        return (z > 0).astype(float)
    elif key == 'tanh':
        return 1 - np.tanh(z) ** 2
    elif key == 'leaky_relu':
        return np.where(z > 0, 1.0, 0.01)
    elif key == 'elu':
        return np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))
    elif key == 'swish':
        s = sigmoid(z); return s + z * s * (1 - s)
    s = sigmoid(z); return s * (1 - s)

# ── Utilitários ────────────────────────────────────────────────────────────────
def _b64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    s = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    plt.close(fig); return s

def _interp_color(t):
    """t∈[0,1]: preto→laranja vibrante para colorir gradiente."""
    r = int(0x0d + t*(0xff-0x0d))
    g = int(0x15 + t*(0x6b-0x15))
    b = int(0x25*(1-t*0.88))
    return f'#{r:02x}{g:02x}{b:02x}'

# ── Diagrama da rede ───────────────────────────────────────────────────────────
NP = {'X':(0.5,0.62),'HA1':(1.9,0.85),'HA2':(1.9,0.38),
      'HB1':(3.3,0.85),'HB2':(3.3,0.38),'Y':(4.7,0.62)}
NC = {'X':'#a07820','HA1':'#2060a0','HA2':'#2060a0',
      'HB1':'#206840','HB2':'#206840','Y':'#802020'}
R  = 0.13

def _base_ax():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 5.4); ax.set_ylim(-0.05, 1.18); ax.axis('off')
    for lx, lb in [(0.5,'Entrada X'),(1.9,'Oculta A'),(3.3,'Oculta B'),(4.7,'Saída ŷ')]:
        ax.text(lx, 1.10, lb, ha='center', fontsize=8.5, color=MUTED, style='italic')
    return fig, ax

def _arrow(ax, src, dst, color, lbl, sgn):
    xs,ys=NP[src]; xe,ye=NP[dst]; dx,dy=xe-xs,ye-ys; d=np.sqrt(dx**2+dy**2)
    ax.annotate("",xy=(xe-(dx/d)*R,ye-(dy/d)*R),xytext=(xs+(dx/d)*R,ys+(dy/d)*R),
                arrowprops=dict(arrowstyle="->",color=color,lw=1.5,alpha=0.85))
    ax.text((xs+xe)/2,(ys+ye)/2+sgn*0.055,lbl,fontsize=6.5,ha='center',color=color,
            fontweight='bold',bbox=dict(boxstyle='round,pad=0.12',fc='#111827',ec='none',alpha=0.92))

def _nodes(ax, vals, delta=False, grad_mags=None):
    mx = max(grad_mags.values()) if grad_mags else 1.0
    mx = max(mx, 1e-12)
    for n,(px,py) in NP.items():
        if grad_mags and n != 'X' and n in grad_mags:
            node_color = _interp_color(abs(grad_mags[n]) / mx)
        else:
            node_color = NC[n]
        ax.add_artist(plt.Circle((px,py),R,color=node_color,ec='#88aacc',lw=1.8,zorder=4))
        top = n if (not delta or n == 'X') else 'δ'
        ax.text(px,py+0.030,top,ha='center',va='center',fontsize=7.5,
                fontweight='bold',zorder=5,color='#cbd5e1')
        vc = ORANGE if (delta and n != 'X') else TEXT
        ax.text(px,py-0.055,f"{vals.get(n,0):.4f}",ha='center',va='center',
                fontsize=7,color=vc,zorder=5)

# ── Textos explicativos de cada ativação ──────────────────────────────────────
_ACT_TEXT = {
    'sigmoid':{
        'desc':(
            "A sigmoide comprime qualquer entrada para o intervalo (0, 1). "
            "Sua derivada pode ser expressa em função da própria saída: "
            "se já calculamos a = σ(z) no forward pass, a derivada é simplesmente a·(1−a). "
            "Isso torna o backpropagation eficiente — não precisamos rearmazenar z."
        ),
        'formula': r"\sigma(z) = \frac{1}{1+e^{-z}} \in (0,\,1)",
        'deriv':   r"\sigma'(z) = \sigma(z)\cdot(1-\sigma(z)) \;\leq\; 0.25",
        'warn':(
            "Atenção: para |z| > 4, σ'(z) ≈ 0 — o gradiente desaparece (vanishing gradient). "
            "Em redes profundas isso impede o aprendizado nas primeiras camadas. "
            "ReLU e Swish foram criadas exatamente para resolver esse problema."
        ),
        'warn_v':'orange',
    },
    'relu':{
        'desc':(
            "A ReLU é a função mais usada em redes profundas modernas. "
            "Para entradas positivas, é simplesmente a identidade — sem saturação. "
            "Para negativas, retorna exatamente zero, criando esparsidade: "
            "muitos neurônios 'silenciam', reduzindo custo computacional."
        ),
        'formula': r"f(z) = \max(0,\, z)",
        'deriv':   r"f'(z) = \begin{cases}0 & z \leq 0 \\ 1 & z > 0\end{cases}",
        'warn':(
            "Risco: 'dying ReLU'. Se z persistir negativo para um neurônio, "
            "δ = 0 e o peso nunca é atualizado — o neurônio 'morre'. "
            "Leaky ReLU e ELU resolvem isso mantendo um gradiente pequeno no lado negativo."
        ),
        'warn_v':'orange',
    },
    'tanh':{
        'desc':(
            "A tangente hiperbólica tem a mesma forma S da sigmoide, mas com saída centrada em zero: "
            "intervalo (−1, 1) em vez de (0, 1). "
            "Ativações centradas em zero facilitam o aprendizado porque os gradientes não ficam "
            "sistematicamente positivos ou negativos, acelerando a convergência. "
            "Por isso a tanh é preferida à sigmoide nas camadas ocultas de RNNs e LSTMs."
        ),
        'formula': r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1,\,1)",
        'deriv':   r"\tanh'(z) = 1 - \tanh^2(z) \;\leq\; 1",
        'warn':(
            "Ainda sofre de vanishing gradient para |z| grande, "
            "mas é menos severo que a sigmoide: derivada máxima = 1 vs 0.25. "
            "Em redes muito profundas (>5 camadas), ReLU e Swish ainda são preferidas."
        ),
        'warn_v':'orange',
    },
    'leaky_relu':{
        'desc':(
            "A Leaky ReLU corrige o maior problema da ReLU padrão: "
            "no lado negativo, em vez de zero, mantém uma inclinação pequena α = 0.01. "
            "Isso garante que o gradiente nunca seja exatamente zero — "
            "neurônios 'mortos' podem se recuperar durante o treinamento. "
            "O valor α é um hiperparâmetro; quando α é aprendido, "
            "a função é chamada de Parametric ReLU (PReLU)."
        ),
        'formula': r"f(z) = \begin{cases}\alpha z & z \leq 0 \\ z & z > 0\end{cases},\quad \alpha=0.01",
        'deriv':   r"f'(z) = \begin{cases}\alpha=0.01 & z \leq 0 \\ 1 & z > 0\end{cases}",
        'warn':(
            "Vantagem: gradiente mínimo = 0.01 (nunca zero). "
            "Limitação: a 'dobra' em z=0 ainda é uma descontinuidade na derivada. "
            "ELU suaviza esse ponto com uma exponencial."
        ),
        'warn_v':'teal',
    },
    'elu':{
        'desc':(
            "A ELU usa uma exponencial suave para z < 0, eliminando a descontinuidade da Leaky ReLU em z = 0. "
            "Isso faz com que a média das ativações fique próxima de zero automaticamente — "
            "propriedade chamada 'auto-normalizante'. "
            "É mais custosa computacionalmente que ReLU por causa da exponencial, "
            "mas converge mais rápido em redes profundas."
        ),
        'formula': r"f(z) = \begin{cases}z & z > 0 \\ \alpha(e^z - 1) & z \leq 0\end{cases},\quad \alpha=1",
        'deriv':   r"f'(z) = \begin{cases}1 & z > 0 \\ \alpha e^z = f(z)+\alpha & z \leq 0\end{cases}",
        'warn':(
            "Nota elegante: a derivada para z ≤ 0 é f(z) + α. "
            "Se já calculamos a saída no forward pass, a derivada é quase grátis — "
            "só uma adição de α."
        ),
        'warn_v':'teal',
    },
    'swish':{
        'desc':(
            "A Swish (ou SiLU) foi descoberta pelo Google Brain via busca automatizada de funções. "
            "É auto-modulada: o sinal z é multiplicado pela sua própria probabilidade de ser relevante σ(z). "
            "Diferente de todas as outras, é não-monotônica: "
            "há um pequeno vale próximo de z = −1.28, depois do qual a função sobe. "
            "É usada em EfficientNet, LLaMA e outros modelos modernos."
        ),
        'formula': r"f(z) = z\cdot\sigma(z) = \frac{z}{1+e^{-z}}",
        'deriv':   r"f'(z) = \sigma(z)\bigl(1 + z(1-\sigma(z))\bigr)",
        'warn':(
            "Propriedade única: não-monótona — cresce, desce levemente perto de z ≈ −1.3, depois cresce. "
            "Isso permite gradientes ligeiramente negativos para entradas levemente negativas, "
            "aumentando empiricamente a capacidade de representação da rede."
        ),
        'warn_v':'teal',
    },
}

# ── Gráficos de ativação ───────────────────────────────────────────────────────
def plot_activation_single(key):
    z = np.linspace(-5, 5, 400)
    f_  = act_fn(z, key)
    df_ = dact(z, key)
    color = ACT_COLORS[key]
    name  = ACT_NAMES[key]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    axes[0].plot(z, f_, color=color, lw=2.5)
    axes[0].axhline(0, color=MUTED, ls='--', lw=0.8)
    axes[0].axvline(0, color=MUTED, ls='--', lw=0.8)
    axes[0].set_title(f'{name}  —  f(z)', fontsize=12, color=TEXT)
    axes[0].set_xlabel('z'); axes[0].set_ylabel('f(z)')

    axes[1].plot(z, df_, color=ORANGE, lw=2.5)
    axes[1].fill_between(z, df_, alpha=0.12, color=ORANGE)
    axes[1].axhline(0, color=MUTED, ls='--', lw=0.8)
    axes[1].set_title("Derivada  f'(z)", fontsize=12, color=TEXT)
    axes[1].set_xlabel('z'); axes[1].set_ylabel("f'(z)")
    fin_df = df_[np.isfinite(df_)]
    if len(fin_df):
        mx_d = float(np.max(fin_df))
        axes[1].annotate(f"máx f'= {mx_d:.3f}", xy=(0, mx_d),
                         xytext=(1.5, mx_d*0.8), color=ORANGE, fontsize=8.5,
                         arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))
    plt.tight_layout(); return _b64(fig)

def plot_activation_comparison(selected_key):
    z     = np.linspace(-4, 4, 400)
    keys  = list(ACT_NAMES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in keys:
        lw    = 3.0 if k == selected_key else 1.4
        alpha = 1.0 if k == selected_key else 0.55
        axes[0].plot(z, act_fn(z,k), color=ACT_COLORS[k], lw=lw, alpha=alpha, label=ACT_NAMES[k])
        axes[1].plot(z, dact(z,k),   color=ACT_COLORS[k], lw=lw, alpha=alpha, label=ACT_NAMES[k])
    for ax in axes:
        ax.axhline(0, color=MUTED, ls='--', lw=0.7)
        ax.axvline(0, color=MUTED, ls='--', lw=0.7)
        ax.set_xlabel('z'); ax.legend(fontsize=7.5, ncol=2)
    axes[0].set_title('Saída  f(z)', fontsize=11, color=TEXT)
    axes[0].set_ylabel('f(z)'); axes[0].set_ylim(-2.2, 5.0)
    axes[1].set_title("Derivada  f'(z)", fontsize=11, color=TEXT)
    axes[1].set_ylabel("f'(z)"); axes[1].set_ylim(-0.15, 1.35)
    plt.suptitle(f'Comparativo — {ACT_NAMES[selected_key]} em destaque', fontsize=11, color=TEXT)
    plt.tight_layout(); return _b64(fig)

# ── Diagramas forward e backprop ──────────────────────────────────────────────
def plot_forward(X, hA, hB, yp, W1, W2, W3, title):
    fig,ax=_base_ax()
    conns=[('X','HA1',f"w={W1[0]:.3f}",+1),('X','HA2',f"w={W1[1]:.3f}",-1),
           ('HA1','HB1',f"w={W2[0,0]:.3f}",+1),('HA1','HB2',f"w={W2[0,1]:.3f}",-1),
           ('HA2','HB1',f"w={W2[1,0]:.3f}",+1),('HA2','HB2',f"w={W2[1,1]:.3f}",-1),
           ('HB1','Y',f"w={W3[0]:.3f}",+1),('HB2','Y',f"w={W3[1]:.3f}",-1)]
    for s,d,lb,sg in conns: _arrow(ax,s,d,TEAL,lb,sg)
    _nodes(ax,{'X':X,'HA1':hA[0],'HA2':hA[1],'HB1':hB[0],'HB2':hB[1],'Y':yp})
    ax.set_title(title,fontsize=10.5,color=TEXT,pad=8)
    plt.tight_layout(); return _b64(fig)

def plot_backprop(X, dW1, dW2, dW3, dhA, dhB, dY, title):
    grad_mags={'Y':abs(dY),'HB1':abs(dhB[0]),'HB2':abs(dhB[1]),
               'HA1':abs(dhA[0]),'HA2':abs(dhA[1])}
    fig,ax=_base_ax()
    conns=[('Y','HB1',f"∇={dW3[0]:.4f}",+1),('Y','HB2',f"∇={dW3[1]:.4f}",-1),
           ('HB1','HA1',f"∇={dW2[0,0]:.4f}",+1),('HB2','HA1',f"∇={dW2[0,1]:.4f}",-1),
           ('HB1','HA2',f"∇={dW2[1,0]:.4f}",+1),('HB2','HA2',f"∇={dW2[1,1]:.4f}",-1),
           ('HA1','X',f"∇={dW1[0]:.4f}",+1),('HA2','X',f"∇={dW1[1]:.4f}",-1)]
    for s,d,lb,sg in conns: _arrow(ax,s,d,ORANGE,lb,sg)
    _nodes(ax,{'X':X,'HA1':dhA[0],'HA2':dhA[1],'HB1':dhB[0],'HB2':dhB[1],'Y':dY},
           delta=True, grad_mags=grad_mags)
    ax.legend(handles=[
        mpatches.Patch(color=_interp_color(1.00), label='|δ| alto'),
        mpatches.Patch(color=_interp_color(0.40), label='|δ| médio'),
        mpatches.Patch(color=_interp_color(0.05), label='|δ| baixo (vanishing)'),
    ], loc='lower right', fontsize=7.5)
    ax.set_title(title,fontsize=10.5,color=TEXT,pad=8)
    plt.tight_layout(); return _b64(fig)

def plot_gradient_flow(dY, dhB, dhA):
    layers = ['Oculta A\n(|δ| médio)', 'Oculta B\n(|δ| médio)', 'Saída\n(|δ_Y|)']
    mags   = [float(np.mean(np.abs(dhA))), float(np.mean(np.abs(dhB))), abs(float(dY))]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = ['#82ca9d', '#6c8ebf', ORANGE]
    bars = ax.barh(layers, mags, color=colors, alpha=0.88, height=0.5)
    mx = max(mags) if max(mags) > 0 else 1.0
    for bar, mag, clr in zip(bars, mags, colors):
        ax.text(mag + mx*0.03, bar.get_y()+bar.get_height()/2,
                f'{mag:.6f}', va='center', ha='left', fontsize=9,
                fontfamily='monospace', color=clr)
    if mags[2] > 1e-12:
        for i, m in enumerate(mags[:2]):
            r = m / mags[2]
            ax.text(mx*1.55, i, f'{r:.3f}× da saída', va='center',
                    fontsize=7.5, color=MUTED, style='italic')
    ax.set_xlabel('Magnitude  |δ|', color=TEXT)
    ax.set_title('Fluxo do Gradiente por Camada — Época 1', color=TEXT, fontsize=11)
    ax.set_xlim(0, mx*2.1); ax.axvline(0, color=MUTED, lw=0.6)
    plt.tight_layout(); return _b64(fig)

def plot_loss_landscape(X, Y, W1_f, W2_f, W3_path, act_key):
    zA_f = X * W1_f; hA_f = act_fn(zA_f, act_key)
    zB_f = np.dot(hA_f, W2_f); hB_f = act_fn(zB_f, act_key)
    path = np.array(W3_path)
    cx,cy = path[-1,0], path[-1,1]
    span  = max(2.0, min(4.0, float(np.std(path[:,0])*4+0.6),
                              float(np.std(path[:,1])*4+0.6)))
    n = 60
    g0 = np.linspace(cx-span, cx+span, n)
    g1 = np.linspace(cy-span, cy+span, n)
    G0,G1 = np.meshgrid(g0, g1)
    zY_grid = hB_f[0]*G0 + hB_f[1]*G1
    E_grid  = 0.5*(Y - sigmoid(zY_grid))**2
    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(E_grid.min(), E_grid.max(), 30)
    cf = ax.contourf(G0, G1, E_grid, levels=levels, cmap='plasma', alpha=0.82)
    ax.contour(G0, G1, E_grid, levels=levels[::4], colors='#1e2d45', linewidths=0.7, alpha=0.7)
    cbar = plt.colorbar(cf, ax=ax); cbar.set_label('Erro MSE', color=TEXT, fontsize=9)
    step_s = max(1, len(path)//60)
    ps = path[::step_s]
    ax.plot(ps[:,0], ps[:,1], 'o-', color='#00d4aa', lw=1.8, ms=2.5, alpha=0.9,
            label='Trajetória W3', zorder=6)
    ax.scatter(path[0,0],  path[0,1],  s=150, color=ORANGE,   zorder=10,
               label=f'Início ({path[0,0]:.3f}, {path[0,1]:.3f})',  marker='s')
    ax.scatter(path[-1,0], path[-1,1], s=180, color='#00ffcc', zorder=10,
               label=f'Final  ({path[-1,0]:.3f}, {path[-1,1]:.3f})', marker='*')
    ax.set_xlabel('W3[0]', color=TEXT); ax.set_ylabel('W3[1]', color=TEXT)
    ax.set_title('Loss Landscape — Superfície do Erro  (W3[0] × W3[1])\n'
                 'W1 e W2 fixos nos valores finais', color=TEXT, fontsize=11)
    ax.legend(fontsize=8); ax.grid(False)
    plt.tight_layout(); return _b64(fig)

def plot_weight_changes(W1_i, W1_f, W2_i, W2_f, W3_i, W3_f):
    names  = ['W1[0]','W1[1]','W2[0,0]','W2[0,1]','W2[1,0]','W2[1,1]','W3[0]','W3[1]']
    before = [W1_i[0],W1_i[1],W2_i[0,0],W2_i[0,1],W2_i[1,0],W2_i[1,1],W3_i[0],W3_i[1]]
    after  = [W1_f[0],W1_f[1],W2_f[0,0],W2_f[0,1],W2_f[1,0],W2_f[1,1],W3_f[0],W3_f[1]]
    delta  = [a-b for a,b in zip(after, before)]
    ypos   = list(range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [TEAL if d>=0 else ORANGE for d in delta]
    axes[0].barh(ypos, delta, color=colors, alpha=0.85, height=0.6)
    axes[0].axvline(0, color=MUTED, lw=1.2)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_xlabel('Δ Peso  (final − inicial)', color=TEXT)
    axes[0].set_title('Variação Total de Cada Peso', color=TEXT, fontsize=11)
    mx = max(abs(d) for d in delta) or 0.01
    for i,(d,c) in enumerate(zip(delta,colors)):
        pad=mx*0.03
        axes[0].text(d+(pad if d>=0 else -pad), i, f'{d:+.4f}',
                     va='center', ha=('left' if d>=0 else 'right'),
                     fontsize=8, color=c, fontfamily='monospace')
    axes[1].scatter(before, ypos, color=MUTED,     s=70, zorder=6, marker='o', label='Inicial')
    axes[1].scatter(after,  ypos, color='#00ffcc', s=70, zorder=6, marker='D', label='Final')
    for i in range(len(names)):
        if abs(delta[i]) > 1e-8:
            axes[1].annotate("", xy=(after[i],i), xytext=(before[i],i),
                             arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                             color=TEAL if delta[i]>=0 else ORANGE))
    axes[1].axvline(0, color=MUTED, lw=0.6, ls='--')
    axes[1].set_yticks(ypos); axes[1].set_yticklabels(names, fontsize=9)
    axes[1].set_xlabel('Valor do Peso', color=TEXT)
    axes[1].set_title('Antes → Depois (todos os pesos)', color=TEXT, fontsize=11)
    axes[1].legend(fontsize=8)
    plt.suptitle('Mudança nos pesos: início → final do treinamento', color=TEXT, fontsize=11, y=1.01)
    plt.tight_layout(); return _b64(fig)

def plot_learning_curve(hist_err, hist_pred, Y, lr, X, epochs, act_name):
    ep = list(range(1, len(hist_err)+1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ep, hist_err, color=ORANGE, lw=2)
    marks = sorted(set([0,min(9,len(hist_err)-1),len(hist_err)//2,len(hist_err)-1]))
    axes[0].scatter([ep[m] for m in marks],[hist_err[m] for m in marks],color='#c0392b',zorder=5,s=55)
    axes[0].set_title('Curva de Aprendizado — MSE', fontsize=11, color=TEXT)
    axes[0].set_xlabel('Épocas'); axes[0].set_ylabel('Erro (MSE)')
    axes[1].plot(ep, hist_pred, color=TEAL, lw=2, label='Predição ŷ')
    axes[1].axhline(Y, color='#27ae60', ls='--', lw=2, label=f'Alvo y = {Y}')
    axes[1].set_title('Convergência da Predição', fontsize=11, color=TEXT)
    axes[1].set_xlabel('Épocas'); axes[1].set_ylabel('ŷ'); axes[1].legend()
    plt.suptitle(f'X={X}, y={Y}, lr={lr}, {epochs} épocas  [{act_name}]', color=TEXT, fontsize=11)
    plt.tight_layout(); return _b64(fig)


# ── Função principal ───────────────────────────────────────────────────────────
def nn_run_all(X_s,Y_s,lr_s,w1_0,w1_1,w2_00,w2_01,w2_10,w2_11,w3_0,w3_1,epochs_s,act_s="sigmoid"):
    try:
        X=float(X_s);Y=float(Y_s);lr=float(lr_s);epochs=int(epochs_s)
        W1=np.array([float(w1_0),float(w1_1)])
        W2=np.array([[float(w2_00),float(w2_01)],[float(w2_10),float(w2_11)]])
        W3=np.array([float(w3_0),float(w3_1)])
        act_key  = act_s if act_s in ACT_NAMES else 'sigmoid'
        act_name = ACT_NAMES[act_key]
        at       = _ACT_TEXT[act_key]

        W1_init=W1.copy(); W2_init=W2.copy(); W3_init=W3.copy()
        W3_path=[W3_init.copy()]
        steps=[]

        # ── Passo 0: Configuração ──────────────────────────────────────────
        steps.append({"title":"Configuração da Rede","sections":[
            {"type":"text","content":
             f"MLP com 4 camadas: Entrada (1 neurônio), Oculta A (2), Oculta B (2) e Saída (1). "
             f"Camadas ocultas usam {act_name}; saída usa sempre Sigmoide para manter ŷ ∈ (0, 1)."},
            {"type":"math","content":
             r"X \xrightarrow{W_1} h_A \xrightarrow{W_2} h_B \xrightarrow{W_3} \hat{y}"},
            {"type":"subtitle","content":"Por que precisamos de pesos iniciais?"},
            {"type":"text","content":
             "Os pesos são os parâmetros que a rede vai aprender. Não podemos começar com todos em zero: "
             "se W=0, todos os neurônios calculam a mesma coisa e o gradiente também é zero — "
             "a rede nunca sairia do lugar (problema da simetria). "
             "Valores diferentes 'quebram' essa simetria e permitem que cada neurônio especialize sua função."},
            {"type":"subtitle","content":"O que cada matriz de pesos conecta"},
            {"type":"table","headers":["Matriz","Dimensão","Conecta","Interpretação"],"rows":[
                ["W1","1 × 2","Entrada → Oculta A",
                 f"W1[0]={w1_0} escala X para hA1;  W1[1]={w1_1} escala X para hA2"],
                ["W2","2 × 2","Oculta A → Oculta B",
                 "Cada coluna combina hA1 e hA2 para produzir um neurônio de hB"],
                ["W3","2 × 1","Oculta B → Saída",
                 f"W3[0]={w3_0} pondera hB1;  W3[1]={w3_1} pondera hB2"],
            ]},
            {"type":"table","headers":["Parâmetro","Valor","Papel"],"rows":[
                ["Entrada X",str(X),"Valor que alimenta a rede"],
                ["Alvo y",str(Y),"Saída que a rede deve aprender a produzir"],
                ["lr",str(lr),"Tamanho do passo. Alto → instável; baixo → lento"],
                ["Épocas",str(epochs),"Repetições do ciclo forward→backprop→atualização"],
                ["Ativação",act_name,"Função usada nas camadas ocultas"],
            ]},
            {"type":"highlight","content":f"Objetivo: dado X = {X}, ajustar W1, W2, W3 até ŷ ≈ {Y}.","variant":"teal"},
        ]})

        # ── Passo 1: Função de Ativação ────────────────────────────────────
        act_table_rows=[]
        for zv in [-4,-2,-1,0,1,2,4]:
            zv_a=np.array([float(zv)])
            act_table_rows.append([str(zv),f"{act_fn(zv_a,act_key)[0]:.4f}",f"{dact(zv_a,act_key)[0]:.4f}"])

        steps.append({"title":f"Função de Ativação: {act_name}","sections":[
            {"type":"text","content":at['desc']},
            {"type":"math","content":at['formula']},
            {"type":"text","content":"Derivada — usada pelo backpropagation para saber o quanto ajustar cada peso:"},
            {"type":"math","content":at['deriv']},
            {"type":"img","content":plot_activation_single(act_key)},
            {"type":"table","headers":["z","f(z)","f'(z)"],"rows":act_table_rows},
            {"type":"highlight","content":at['warn'],"variant":at['warn_v']},
        ]})

        # ── Passo 2: Comparativo ───────────────────────────────────────────
        steps.append({"title":"Comparativo — Todas as Funções de Ativação","sections":[
            {"type":"text","content":
             f"Você escolheu {act_name} (linha destacada). "
             "Compare as saídas e, principalmente, as derivadas — "
             "são elas que determinam a velocidade e estabilidade do aprendizado."},
            {"type":"img","content":plot_activation_comparison(act_key)},
            {"type":"table","headers":["Função","Saída","Deriv. máx","Risco principal","Uso típico"],"rows":[
                ["Sigmoide","(0, 1)","0.25","Vanishing gradient severo","Saída binária"],
                ["ReLU","[0, ∞)","1","Dying ReLU","CNNs, redes densas"],
                ["Tanh","(−1, 1)","1.0","Vanishing gradient (leve)","RNNs, LSTMs"],
                ["Leaky ReLU","(−∞, ∞)","1","Kink em z=0","Substituto de ReLU"],
                ["ELU","(−1, ∞)","1","Exp. custosa","Redes profundas"],
                ["Swish","(≈−0.28, ∞)","~1.1","Não-monotônica","GPT, EfficientNet"],
            ]},
            {"type":"highlight",
             "content":"Regra geral: sigmoide/tanh para redes pequenas ou saídas; "
                       "ReLU e variantes para camadas ocultas de redes profundas; "
                       "Swish/ELU quando ReLU não converge bem.",
             "variant":"teal"},
        ]})

        # ── Passo 3: Forward Pass Época 1 ─────────────────────────────────
        zA=X*W1; hA=act_fn(zA,act_key)
        zB=np.dot(hA,W2); hB=act_fn(zB,act_key)
        zY=np.dot(hB,W3); yp=sigmoid(zY)
        err1=0.5*(Y-yp)**2

        steps.append({"title":"Forward Pass — Época 1","sections":[
            {"type":"text","content":
             f"Cada camada faz uma combinação linear e aplica {act_name} "
             "(ocultas) ou Sigmoide (saída). Veja cada número explicitamente."},
            {"type":"img","content":plot_forward(X,hA,hB,yp,W1,W2,W3,
                f"Época 1 — Forward Pass  (setas: pesos;  nós: ativações)")},
            {"type":"subtitle","content":f"① Oculta A  —  z_A = X · W1,  h_A = {act_name}(z_A)"},
            {"type":"math","content":
             r"z_{A_1}=X\times W_1[0]="+ f"{X}\\times{W1[0]}={zA[0]:.4f}"},
            {"type":"math","content":
             r"z_{A_2}=X\times W_1[1]="+ f"{X}\\times{W1[1]}={zA[1]:.4f}"},
            {"type":"math","content":
             f"h_A=[{act_name}({zA[0]:.4f}),\\ {act_name}({zA[1]:.4f})]=[{hA[0]:.4f},\\ {hA[1]:.4f}]"},
            {"type":"subtitle","content":f"② Oculta B  —  z_B = h_A · W2,  h_B = {act_name}(z_B)"},
            {"type":"math","content":
             r"z_{B_1}=h_{A_1}\times W_2[0,0]+h_{A_2}\times W_2[1,0]="
             +f"{hA[0]:.4f}\\times{W2[0,0]}+{hA[1]:.4f}\\times{W2[1,0]}={zB[0]:.4f}"},
            {"type":"math","content":
             r"z_{B_2}=h_{A_1}\times W_2[0,1]+h_{A_2}\times W_2[1,1]="
             +f"{hA[0]:.4f}\\times{W2[0,1]}+{hA[1]:.4f}\\times{W2[1,1]}={zB[1]:.4f}"},
            {"type":"math","content":
             f"h_B=[{act_name}({zB[0]:.4f}),\\ {act_name}({zB[1]:.4f})]=[{hB[0]:.4f},\\ {hB[1]:.4f}]"},
            {"type":"subtitle","content":"③ Saída  —  z_Y = h_B · W3,  ŷ = σ(z_Y)"},
            {"type":"math","content":
             r"z_Y=h_{B_1}\times W_3[0]+h_{B_2}\times W_3[1]="
             +f"{hB[0]:.4f}\\times{W3[0]}+{hB[1]:.4f}\\times{W3[1]}={zY:.4f}"},
            {"type":"math","content":
             r"\hat{y}=\sigma(z_Y)=\frac{1}{1+e^{-("
             +f"{zY:.4f}"+r")}}="+f"{yp:.4f}"},
            {"type":"highlight","content":f"Predição: ŷ = {yp:.4f}  |  Alvo: y = {Y}  |  Erro: E = {err1:.6f}","variant":"orange"},
        ]})

        # ── Passo 4: Função de Custo ───────────────────────────────────────
        steps.append({"title":"Função de Custo — Época 1","sections":[
            {"type":"text","content":
             "Precisamos de um número que meça o quão errada está a rede. "
             "Usamos o MSE com fator ½ para simplificar o gradiente: "
             "ao derivar, o expoente 2 cancela o ½."},
            {"type":"math","content":r"E = \frac{1}{2}(y - \hat{y})^2"},
            {"type":"subtitle","content":"Conta com os valores da Época 1"},
            {"type":"math","content":
             r"E=\frac{1}{2}("+f"{Y}-{yp:.4f}"+r")^2"
             +r"=\frac{1}{2}\times("+f"{Y-yp:.4f}"+r")^2"
             +r"=\frac{1}{2}\times"+f"{(Y-yp)**2:.6f}"
             +r"="+f"{err1:.6f}"},
            {"type":"text","content":
             "Por que elevar ao quadrado? "
             "(1) erros positivos e negativos contribuem igualmente; "
             "(2) erros grandes são penalizados desproporcionalmente — 0.1²=0.01, mas 0.5²=0.25."},
            {"type":"highlight","content":
             f"A rede prevê {yp:.4f}, deve prever {Y}. "
             "Backpropagation vai calcular em qual direção mover cada peso para reduzir E.",
             "variant":"orange"},
        ]})

        # ── Passo 5: Backpropagation Época 1 ──────────────────────────────
        W1b=W1.copy(); W2b=W2.copy(); W3b=W3.copy()
        dsY=d_sig_out(yp); dY_v=(yp-Y)*dsY; dW3=dY_v*hB
        dsB=dact(zB,act_key); dhB=dY_v*W3*dsB; dW2=np.outer(hA,dhB)
        dsA=dact(zA,act_key); dhA=np.dot(dhB,W2.T)*dsA; dW1=dhA*X
        W3-=lr*dW3; W2-=lr*dW2; W1-=lr*dW1; W3_path.append(W3.copy())

        steps.append({"title":"Backpropagation — Época 1","sections":[
            {"type":"text","content":
             "A regra da cadeia propaga o erro de trás para frente. "
             "A intensidade da cor de cada nó no diagrama indica a magnitude de δ: "
             "nós brilhantes (laranja) = gradiente forte; "
             "nós escuros (quase pretos) = gradiente próximo de zero (vanishing gradient)."},
            {"type":"img","content":plot_backprop(X,dW1,dW2,dW3,dhA,dhB,dY_v,
                "Época 1 — Backpropagation  (cor dos nós = magnitude |δ|)")},
            {"type":"subtitle","content":"① δ_Y  (saída usa Sigmoide)"},
            {"type":"math","content":
             r"\sigma'(\hat{y})=\hat{y}(1-\hat{y})="
             +f"{yp:.4f}\\times{1-yp:.4f}={dsY:.4f}"},
            {"type":"math","content":
             r"\delta_Y=(\hat{y}-y)\cdot\sigma'(\hat{y})="
             +f"({yp:.4f}-{Y})\\times{dsY:.4f}={dY_v:.6f}"},
            {"type":"subtitle","content":f"② ∇W3 = δ_Y · h_B"},
            {"type":"math","content":r"\nabla W_3=["+f"{dW3[0]:.4f},\\ {dW3[1]:.4f}"+r"]"},
            {"type":"subtitle","content":f"③ δ_hB — derivada de {act_name} em z_B"},
            {"type":"math","content":
             f"f'(z_{{B_1}})={dsB[0]:.4f},\\quad f'(z_{{B_2}})={dsB[1]:.4f}"},
            {"type":"math","content":
             r"\delta_{h_{B_1}}=\delta_Y\times W_3[0]\times f'(z_{B_1})="
             +f"{dY_v:.4f}\\times{W3b[0]:.4f}\\times{dsB[0]:.4f}={dhB[0]:.6f}"},
            {"type":"math","content":
             r"\delta_{h_{B_2}}=\delta_Y\times W_3[1]\times f'(z_{B_2})="
             +f"{dY_v:.4f}\\times{W3b[1]:.4f}\\times{dsB[1]:.4f}={dhB[1]:.6f}"},
            {"type":"subtitle","content":"④ ∇W2 = h_A^T ⊗ δ_hB"},
            {"type":"math","content":
             r"\nabla W_2=\begin{bmatrix}"
             +f"{hA[0]:.3f}\\times{dhB[0]:.4f}&{hA[0]:.3f}\\times{dhB[1]:.4f}"
             +r"\\"+f"{hA[1]:.3f}\\times{dhB[0]:.4f}&{hA[1]:.3f}\\times{dhB[1]:.4f}"
             +r"\end{bmatrix}=\begin{bmatrix}"
             +f"{dW2[0,0]:.4f}&{dW2[0,1]:.4f}"+r"\\"+f"{dW2[1,0]:.4f}&{dW2[1,1]:.4f}"
             +r"\end{bmatrix}"},
            {"type":"subtitle","content":f"⑤ δ_hA e ∇W1"},
            {"type":"math","content":
             f"f'(z_{{A_1}})={dsA[0]:.4f},\\quad f'(z_{{A_2}})={dsA[1]:.4f}"},
            {"type":"math","content":
             r"\delta_{h_{A_1}}=(\delta_{h_{B_1}}\cdot W_2[0,0]+\delta_{h_{B_2}}\cdot W_2[0,1])\cdot f'(z_{A_1})="
             +f"{dhA[0]:.6f}"},
            {"type":"math","content":
             r"\nabla W_1=[\delta_{h_{A_1}}\times X,\ \delta_{h_{A_2}}\times X]="
             +f"[{dW1[0]:.6f},\\ {dW1[1]:.6f}"+r"]"},
            {"type":"subtitle","content":f"⑥ Atualização  W ← W − {lr} · ∇W"},
            {"type":"table","headers":["Peso","Antes","−lr·∇W","Depois"],"rows":[
                ["W3[0]",f"{W3b[0]:.4f}",f"−{lr}×{dW3[0]:.4f}={-lr*dW3[0]:.4f}",f"{W3[0]:.4f}"],
                ["W3[1]",f"{W3b[1]:.4f}",f"−{lr}×{dW3[1]:.4f}={-lr*dW3[1]:.4f}",f"{W3[1]:.4f}"],
                ["W2[0,0]",f"{W2b[0,0]:.4f}",f"−{lr}×{dW2[0,0]:.4f}={-lr*dW2[0,0]:.4f}",f"{W2[0,0]:.4f}"],
                ["W2[0,1]",f"{W2b[0,1]:.4f}",f"−{lr}×{dW2[0,1]:.4f}={-lr*dW2[0,1]:.4f}",f"{W2[0,1]:.4f}"],
                ["W2[1,0]",f"{W2b[1,0]:.4f}",f"−{lr}×{dW2[1,0]:.4f}={-lr*dW2[1,0]:.4f}",f"{W2[1,0]:.4f}"],
                ["W2[1,1]",f"{W2b[1,1]:.4f}",f"−{lr}×{dW2[1,1]:.4f}={-lr*dW2[1,1]:.4f}",f"{W2[1,1]:.4f}"],
                ["W1[0]",f"{W1b[0]:.4f}",f"−{lr}×{dW1[0]:.4f}={-lr*dW1[0]:.4f}",f"{W1[0]:.4f}"],
                ["W1[1]",f"{W1b[1]:.4f}",f"−{lr}×{dW1[1]:.4f}={-lr*dW1[1]:.4f}",f"{W1[1]:.4f}"],
            ]},
            {"type":"subtitle","content":"Fluxo do Gradiente por Camada"},
            {"type":"text","content":
             "O gráfico abaixo mostra como a magnitude de δ diminui ao recuar pelas camadas. "
             "Barra muito menor nas camadas iniciais = vanishing gradient em ação."},
            {"type":"img","content":plot_gradient_flow(dY_v,dhB,dhA)},
        ]})

        # ── Passo 6: Época 2 ───────────────────────────────────────────────
        zA2=X*W1; hA2=act_fn(zA2,act_key); zB2=np.dot(hA2,W2); hB2=act_fn(zB2,act_key)
        zY2=np.dot(hB2,W3); yp2=sigmoid(zY2); err2=0.5*(Y-yp2)**2
        W3b2=W3.copy()
        dY2=(yp2-Y)*d_sig_out(yp2); dW3_2=dY2*hB2
        dhB2=dY2*W3*dact(zB2,act_key); dW2_2=np.outer(hA2,dhB2)
        dhA2=np.dot(dhB2,W2.T)*dact(zA2,act_key); dW1_2=dhA2*X
        W3-=lr*dW3_2; W2-=lr*dW2_2; W1-=lr*dW1_2; W3_path.append(W3.copy())

        steps.append({"title":"Época 2 — Ciclo Completo","sections":[
            {"type":"text","content":"Mesmo ciclo, pesos atualizados. O erro deve ser menor."},
            {"type":"subtitle","content":"Forward Pass"},
            {"type":"img","content":plot_forward(X,hA2,hB2,yp2,W1,W2,W3,"Época 2 — pesos ajustados pela época 1")},
            {"type":"math","content":
             r"h_A=["+f"{hA2[0]:.4f},\\ {hA2[1]:.4f}"+r"],\quad h_B=["+f"{hB2[0]:.4f},\\ {hB2[1]:.4f}"+r"]"},
            {"type":"math","content":
             r"\hat{y}="+f"{yp2:.4f}"+r",\quad E="+f"{err2:.6f}"
             +r"\quad(\text{época 1: }"+f"{err1:.6f}"+r")"},
            {"type":"highlight","content":f"Erro:  {err1:.6f} → {err2:.6f}  ({(1-err2/err1)*100:.1f}% menor)","variant":"teal"},
            {"type":"subtitle","content":"Backpropagation — Época 2"},
            {"type":"img","content":plot_backprop(X,dW1_2,dW2_2,dW3_2,dhA2,dhB2,dY2,"Época 2 — Backpropagation")},
            {"type":"math","content":
             r"\delta_Y="+f"{dY2:.6f}"+r"\quad(\text{época 1: }"+f"{dY_v:.6f}"+r")\;\Rightarrow\text{ sinal de erro diminuiu}"},
        ]})

        # ── Passo 7: Treinamento completo ──────────────────────────────────
        hist_e=[err1,err2]; hist_p=[yp,yp2]
        for ep in range(3, epochs+1):
            zA_=X*W1; hA_=act_fn(zA_,act_key); zB_=np.dot(hA_,W2); hB_=act_fn(zB_,act_key)
            zY_=np.dot(hB_,W3); yp_=sigmoid(zY_); e_=0.5*(Y-yp_)**2
            hist_e.append(e_); hist_p.append(yp_)
            dY_=(yp_-Y)*d_sig_out(yp_); dW3_=dY_*hB_
            dhB_=dY_*W3*dact(zB_,act_key); dW2_=np.outer(hA_,dhB_)
            dhA_=np.dot(dhB_,W2.T)*dact(zA_,act_key); dW1_=dhA_*X
            W3-=lr*dW3_; W2-=lr*dW2_; W1-=lr*dW1_; W3_path.append(W3.copy())

        marks=sorted(set([max(0,x) for x in [0,9,epochs//4-1,epochs//2-1,epochs-1]]))
        rows=[[str(m+1),f"{hist_p[m]:.4f}",f"{hist_e[m]:.6f}"] for m in marks]
        steps.append({"title":f"Treinamento Completo — {epochs} Épocas","sections":[
            {"type":"text","content":
             f"As épocas 1 e 2 foram detalhadas. Aqui executamos até a época {epochs}."},
            {"type":"img","content":plot_learning_curve(hist_e,hist_p,Y,lr,X,epochs,act_name)},
            {"type":"table","headers":["Época","Predição ŷ","Erro (MSE)"],"rows":rows},
            {"type":"highlight","content":
             f"Redução total: {hist_e[0]:.6f} → {hist_e[-1]:.6f}  "
             f"({(1-hist_e[-1]/hist_e[0])*100:.1f}% de melhora em {epochs} épocas)",
             "variant":"teal"},
        ]})

        # ── Passo 8: Loss Landscape ────────────────────────────────────────
        steps.append({"title":"Loss Landscape — Superfície do Erro","sections":[
            {"type":"text","content":
             "A superfície do erro mostra o valor de E para todas as combinações de W3[0] e W3[1] "
             "(W1 e W2 fixos nos valores finais). "
             "A trajetória teal é o caminho percorrido pela descida do gradiente. "
             "Quadrado laranja = início; estrela verde = fim."},
            {"type":"img","content":plot_loss_landscape(X,Y,W1,W2,W3_path,act_key)},
            {"type":"text","content":
             "Regiões escuras (roxo) = erro baixo. "
             "Regiões claras (amarelo) = erro alto. "
             "A trajetória deveria terminar numa região escura se o treinamento convergiu. "
             "Em redes reais com milhões de parâmetros essa superfície é um hipervolume "
             "impossível de visualizar — mas a intuição é a mesma."},
            {"type":"highlight","content":
             "Se a trajetória terminar num platô (região de baixo gradiente sem ser o mínimo), "
             "tente aumentar épocas, ajustar lr, ou mudar a função de ativação.",
             "variant":"teal"},
        ]})

        # ── Passo 9: Variação dos Pesos ────────────────────────────────────
        steps.append({"title":"Variação Total dos Pesos — Início → Final","sections":[
            {"type":"text","content":
             f"Compara os pesos iniciais com os valores aprendidos após {epochs} épocas. "
             "Barras teal = peso aumentou; barras laranja = peso diminuiu. "
             "A seta no painel direito aponta na direção da mudança."},
            {"type":"img","content":plot_weight_changes(W1_init,W1,W2_init,W2,W3_init,W3)},
            {"type":"table","headers":["Peso","Inicial","Final","Variação"],"rows":[
                ["W1[0]",  f"{W1_init[0]:.4f}",f"{W1[0]:.4f}",  f"{W1[0]-W1_init[0]:+.4f}"],
                ["W1[1]",  f"{W1_init[1]:.4f}",f"{W1[1]:.4f}",  f"{W1[1]-W1_init[1]:+.4f}"],
                ["W2[0,0]",f"{W2_init[0,0]:.4f}",f"{W2[0,0]:.4f}",f"{W2[0,0]-W2_init[0,0]:+.4f}"],
                ["W2[0,1]",f"{W2_init[0,1]:.4f}",f"{W2[0,1]:.4f}",f"{W2[0,1]-W2_init[0,1]:+.4f}"],
                ["W2[1,0]",f"{W2_init[1,0]:.4f}",f"{W2[1,0]:.4f}",f"{W2[1,0]-W2_init[1,0]:+.4f}"],
                ["W2[1,1]",f"{W2_init[1,1]:.4f}",f"{W2[1,1]:.4f}",f"{W2[1,1]-W2_init[1,1]:+.4f}"],
                ["W3[0]",  f"{W3_init[0]:.4f}",f"{W3[0]:.4f}",  f"{W3[0]-W3_init[0]:+.4f}"],
                ["W3[1]",  f"{W3_init[1]:.4f}",f"{W3[1]:.4f}",  f"{W3[1]-W3_init[1]:+.4f}"],
            ]},
            {"type":"highlight","content":
             "Pesos com maior variação foram os mais influentes na redução do erro. "
             "Pesos quase estáticos podem indicar vanishing gradient ou baixa influência no exemplo atual.",
             "variant":"orange"},
        ]})

        # ── Passo 10: Estado Final ─────────────────────────────────────────
        zA_f=X*W1; hA_f=act_fn(zA_f,act_key); zB_f=np.dot(hA_f,W2); hB_f=act_fn(zB_f,act_key)
        zY_f=np.dot(hB_f,W3); yp_f=sigmoid(zY_f)
        steps.append({"title":"Estado Final da Rede","sections":[
            {"type":"text","content":f"Após {epochs} épocas com {act_name} nas camadas ocultas."},
            {"type":"img","content":plot_forward(X,hA_f,hB_f,yp_f,W1,W2,W3,
                f"Época {epochs} — Predição: {yp_f:.4f}  |  Alvo: {Y}")},
            {"type":"math","content":
             r"\hat{y}_{final}="+f"{yp_f:.6f}"
             +r"\;\longrightarrow\; y="+f"{Y}"
             +r"\quad(E_{final}="+f"{hist_e[-1]:.8f}"+r")"},
            {"type":"highlight","content":
             f"Evolução:  ŷ = {hist_p[0]:.4f}  (época 1)  →  {yp_f:.4f}  (época {epochs})   alvo = {Y}",
             "variant":"teal"},
        ]})

        summary={
            "X":X,"Y":Y,"lr":lr,"epochs":epochs,"act":act_name,
            "w1_init":[float(w1_0),float(w1_1)],
            "w2_init":[[float(w2_00),float(w2_01)],[float(w2_10),float(w2_11)]],
            "w3_init":[float(w3_0),float(w3_1)],
            "yp_f":float(yp_f),"err_f":float(hist_e[-1]),
        }
        return json.dumps({"ok":True,"steps":steps,"summary":summary})
    except Exception as e:
        import traceback
        return json.dumps({"ok":False,"error":str(e),"tb":traceback.format_exc()})
