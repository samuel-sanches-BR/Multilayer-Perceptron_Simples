import numpy as np
import matplotlib.pyplot as plt
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

TEAL = '#00d4aa'; ORANGE = '#ff6b35'; MUTED = '#64748b'; TEXT = '#e2e8f0'

def sigmoid(z):  return 1 / (1 + np.exp(-z))
def d_sig(a):    return a * (1 - a)

def _b64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    s = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return s

# ── Network diagram helpers ────────────────────────────────────────────────────
NP = {
    'X':   (0.5, 0.62),
    'HA1': (1.9, 0.85), 'HA2': (1.9, 0.38),
    'HB1': (3.3, 0.85), 'HB2': (3.3, 0.38),
    'Y':   (4.7, 0.62)
}
NC = {
    'X': '#a07820', 'HA1': '#2060a0', 'HA2': '#2060a0',
    'HB1': '#206840', 'HB2': '#206840', 'Y': '#802020'
}
R = 0.13

def _base_ax():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 5.4); ax.set_ylim(-0.05, 1.18); ax.axis('off')
    for lx, lb in [(0.5,'Entrada X'),(1.9,'Oculta A'),(3.3,'Oculta B'),(4.7,'Saída ŷ')]:
        ax.text(lx, 1.10, lb, ha='center', fontsize=8.5, color=MUTED, style='italic')
    return fig, ax

def _arrow(ax, src, dst, color, lbl, sgn):
    xs, ys = NP[src]; xe, ye = NP[dst]
    dx, dy = xe-xs, ye-ys; d = np.sqrt(dx**2+dy**2)
    ax.annotate("", xy=(xe-(dx/d)*R, ye-(dy/d)*R), xytext=(xs+(dx/d)*R, ys+(dy/d)*R),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.85))
    ax.text((xs+xe)/2, (ys+ye)/2+sgn*0.055, lbl, fontsize=6.5, ha='center', color=color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', fc='#111827', ec='none', alpha=0.92))

def _nodes(ax, vals, delta=False):
    for n, (px, py) in NP.items():
        ax.add_artist(plt.Circle((px, py), R, color=NC[n], ec='#556', lw=1.5, zorder=4))
        top = n if (not delta or n == 'X') else 'δ'
        ax.text(px, py+0.030, top, ha='center', va='center', fontsize=7.5,
                fontweight='bold', zorder=5, color='#cbd5e1')
        vc = ORANGE if (delta and n != 'X') else TEXT
        ax.text(px, py-0.055, f"{vals.get(n, 0):.4f}", ha='center', va='center',
                fontsize=7, color=vc, zorder=5)

def plot_sigmoid():
    z = np.linspace(-6, 6, 200); a = sigmoid(z)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(z, a, color=TEAL, lw=2.5)
    axes[0].axhline(0.5, color=MUTED, ls='--', lw=1, label='y = 0.5')
    axes[0].axvline(0, color=MUTED, ls='--', lw=1, label='z = 0')
    axes[0].set_title('Sigmoide  σ(z)', fontsize=12, color=TEXT)
    axes[0].set_xlabel('z'); axes[0].set_ylabel('σ(z)'); axes[0].set_ylim(-0.05,1.05); axes[0].legend()
    axes[1].plot(z, d_sig(a), color=ORANGE, lw=2.5)
    axes[1].set_title("Derivada  σ'(z)", fontsize=12, color=TEXT)
    axes[1].set_xlabel('z'); axes[1].set_ylabel("σ'(z)")
    plt.tight_layout()
    return _b64(fig)

def plot_forward(X, hA, hB, yp, W1, W2, W3, title):
    fig, ax = _base_ax()
    conns = [
        ('X','HA1', f"w={W1[0]:.3f}", +1), ('X','HA2', f"w={W1[1]:.3f}", -1),
        ('HA1','HB1', f"w={W2[0,0]:.3f}", +1), ('HA1','HB2', f"w={W2[0,1]:.3f}", -1),
        ('HA2','HB1', f"w={W2[1,0]:.3f}", +1), ('HA2','HB2', f"w={W2[1,1]:.3f}", -1),
        ('HB1','Y', f"w={W3[0]:.3f}", +1), ('HB2','Y', f"w={W3[1]:.3f}", -1),
    ]
    for s, d, lb, sg in conns: _arrow(ax, s, d, TEAL, lb, sg)
    _nodes(ax, {'X':X,'HA1':hA[0],'HA2':hA[1],'HB1':hB[0],'HB2':hB[1],'Y':yp})
    ax.set_title(title, fontsize=10.5, color=TEXT, pad=8)
    plt.tight_layout(); return _b64(fig)

def plot_backprop(X, dW1, dW2, dW3, dhA, dhB, dY, title):
    fig, ax = _base_ax()
    conns = [
        ('Y','HB1', f"∇={dW3[0]:.4f}", +1), ('Y','HB2', f"∇={dW3[1]:.4f}", -1),
        ('HB1','HA1', f"∇={dW2[0,0]:.4f}", +1), ('HB2','HA1', f"∇={dW2[0,1]:.4f}", -1),
        ('HB1','HA2', f"∇={dW2[1,0]:.4f}", +1), ('HB2','HA2', f"∇={dW2[1,1]:.4f}", -1),
        ('HA1','X', f"∇={dW1[0]:.4f}", +1), ('HA2','X', f"∇={dW1[1]:.4f}", -1),
    ]
    for s, d, lb, sg in conns: _arrow(ax, s, d, ORANGE, lb, sg)
    _nodes(ax, {'X':X,'HA1':dhA[0],'HA2':dhA[1],'HB1':dhB[0],'HB2':dhB[1],'Y':dY}, delta=True)
    ax.set_title(title, fontsize=10.5, color=TEXT, pad=8)
    plt.tight_layout(); return _b64(fig)

def plot_learning_curve(hist_err, hist_pred, Y, lr, X, epochs):
    ep = list(range(1, len(hist_err)+1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ep, hist_err, color=ORANGE, lw=2)
    marks = sorted(set([0, min(9,len(hist_err)-1), len(hist_err)//2, len(hist_err)-1]))
    axes[0].scatter([ep[m] for m in marks], [hist_err[m] for m in marks], color='#c0392b', zorder=5, s=55)
    axes[0].set_title('Curva de Aprendizado — Erro MSE', fontsize=11, color=TEXT)
    axes[0].set_xlabel('Épocas'); axes[0].set_ylabel('Erro (MSE)')
    axes[1].plot(ep, hist_pred, color=TEAL, lw=2, label='Predição ŷ')
    axes[1].axhline(Y, color='#27ae60', ls='--', lw=2, label=f'Alvo y = {Y}')
    axes[1].set_title('Predição convergindo para o alvo', fontsize=11, color=TEXT)
    axes[1].set_xlabel('Épocas'); axes[1].set_ylabel('ŷ'); axes[1].legend()
    plt.suptitle(f'Treinamento: X={X}, alvo={Y}, lr={lr}, {epochs} épocas', color=TEXT, fontsize=11)
    plt.tight_layout(); return _b64(fig)

def plot_taxonomy():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    fig.patch.set_facecolor('#111827')
    for ax in axes:
        ax.set_facecolor('#0a0e1a'); ax.axis('off')

    # MLP
    ax = axes[0]
    ax.set_xlim(0,4); ax.set_ylim(0,5)
    ax.set_title('MLP  (nossa rede ✅)', fontsize=12, color=TEAL, pad=8)
    mlp = [(1,[2.5]),(2,[3.3,1.7]),(3,[3.3,1.7]),(4,[2.5])]
    pos = []
    for lx, ys in mlp:
        layer = []
        for y in ys:
            ax.add_artist(plt.Circle((lx,y),0.28,color='#2060a0',ec=TEAL,lw=1.5,zorder=4))
            layer.append((lx,y))
        pos.append(layer)
    for l in range(len(pos)-1):
        for x0,y0 in pos[l]:
            for x1,y1 in pos[l+1]:
                ax.annotate("",xy=(x1-0.28,y1),xytext=(x0+0.28,y0),
                            arrowprops=dict(arrowstyle="->",color=TEAL,lw=1.2))
    for lx,lb in [(1,'Entrada'),(2,'Oculta A'),(3,'Oculta B'),(4,'Saída')]:
        ax.text(lx,0.45,lb,ha='center',fontsize=7.5,color=MUTED,style='italic')
    ax.text(2.5,4.7,'Conexões densas\ntodos → todos',ha='center',fontsize=9,color=TEAL,
            bbox=dict(boxstyle='round,pad=0.3',fc='#111827',ec=TEAL,alpha=0.9))
    ax.text(2.5,0.12,'Dados tabulares, classificação',ha='center',fontsize=8,color=MUTED)

    # CNN
    ax = axes[1]
    ax.set_xlim(0,5); ax.set_ylim(0,5)
    ax.set_title('CNN  (Convolucional)', fontsize=12, color='#2ecc71', pad=8)
    for i in range(4):
        for j in range(4):
            ax.add_artist(plt.Rectangle((0.3+j*0.4,2.8+i*0.4),0.36,0.36,
                color='#1a3a2a',ec='#2ecc71',lw=1,zorder=3))
    for di in range(2):
        for dj in range(2):
            ax.add_artist(plt.Rectangle((0.3+dj*0.4,2.8+(di+2)*0.4),0.36,0.36,
                color=ORANGE,ec='#ff4000',lw=2,zorder=5))
    ax.text(0.94,2.62,'Entrada (imagem)',ha='center',fontsize=8,color=MUTED,style='italic')
    ax.text(2.15,3.9,'Filtro\n(kernel)',ha='left',fontsize=8.5,color=ORANGE,
            bbox=dict(boxstyle='round,pad=0.3',fc='#111827',ec=ORANGE,alpha=0.9))
    for i in range(3):
        for j in range(3):
            ax.add_artist(plt.Rectangle((0.4+j*0.48,1.3+i*0.48),0.42,0.42,
                color='#0d2b1e',ec='#2ecc71',lw=1.2,zorder=3))
    ax.text(0.94,1.1,'Feature Map',ha='center',fontsize=8,color=MUTED,style='italic')
    ax.annotate("",xy=(0.94,2.77),xytext=(0.94,2.63),arrowprops=dict(arrowstyle="->",color='#2ecc71',lw=1.5))
    ax.text(2.5,4.7,'Pesos\ncompartilhados',ha='center',fontsize=9,color='#2ecc71',
            bbox=dict(boxstyle='round,pad=0.3',fc='#111827',ec='#2ecc71',alpha=0.9))
    ax.text(2.5,0.12,'Imagens, vídeo, sinais 1D',ha='center',fontsize=8,color=MUTED)

    # RNN
    ax = axes[2]
    ax.set_xlim(0,5); ax.set_ylim(0,5)
    ax.set_title('RNN  (Recorrente)', fontsize=12, color='#f1c40f', pad=8)
    steps_rnn = [('t-1',1.0),('t',2.5),('t+1',4.0)]
    for lb,xc in steps_rnn:
        ax.add_artist(plt.Circle((xc,1.2),0.28,color='#7a5a00',ec='#f1c40f',lw=1.5,zorder=4))
        ax.add_artist(plt.Circle((xc,2.8),0.32,color='#7a5a00',ec='#f1c40f',lw=1.5,zorder=4))
        ax.add_artist(plt.Circle((xc,4.2),0.28,color='#7a0000',ec='#f1c40f',lw=1.5,zorder=4))
        for txt,yc in [(f'x({lb})',1.2),(f'h({lb})',2.8),(f'ŷ({lb})',4.2)]:
            ax.text(xc,yc,txt,ha='center',va='center',fontsize=6.5,color='#cbd5e1',fontweight='bold',zorder=5)
        ax.annotate("",xy=(xc,2.46),xytext=(xc,1.50),arrowprops=dict(arrowstyle="->",color='#f1c40f',lw=1.8))
        ax.annotate("",xy=(xc,3.90),xytext=(xc,3.14),arrowprops=dict(arrowstyle="->",color='#f1c40f',lw=1.8))
    for (_,x0),(_,x1) in zip(steps_rnn[:-1],steps_rnn[1:]):
        ax.annotate("",xy=(x1-0.32,2.8),xytext=(x0+0.32,2.8),arrowprops=dict(arrowstyle="->",color='crimson',lw=2.5))
    ax.text(2.5,3.28,'memória\n(estado oculto)',ha='center',fontsize=9,color='crimson',fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',fc='#111827',ec='crimson',alpha=0.9))
    ax.text(2.5,0.12,'Texto, fala, séries temporais',ha='center',fontsize=8,color=MUTED)

    plt.suptitle('Taxonomia: MLP  ·  CNN  ·  RNN', fontsize=13, color=TEXT, y=1.01)
    plt.tight_layout(); return _b64(fig)


# ── Main function ──────────────────────────────────────────────────────────────

def nn_run_all(X_s,Y_s,lr_s,w1_0,w1_1,w2_00,w2_01,w2_10,w2_11,w3_0,w3_1,epochs_s):
    try:
        X=float(X_s); Y=float(Y_s); lr=float(lr_s); epochs=int(epochs_s)
        W1=np.array([float(w1_0),float(w1_1)])
        W2=np.array([[float(w2_00),float(w2_01)],[float(w2_10),float(w2_11)]])
        W3=np.array([float(w3_0),float(w3_1)])

        steps=[]

        # ── Passo 0: Configuração ────────────────────────────────────────
        steps.append({
            "title": "Configuração da Rede",
            "sections": [
                {"type":"text","content":"Arquitetura: X (1 neurônio) → Oculta A (2) → Oculta B (2) → Saída (1). Apenas a camada de saída (readout) é treinada por descida do gradiente."},
                {"type":"table","headers":["Parâmetro","Valor"],"rows":[
                    ["Entrada X", str(X)], ["Alvo y", str(Y)],
                    ["Taxa de aprendizado (lr)", str(lr)], ["Épocas", str(epochs)],
                    ["W1", f"[{W1[0]}, {W1[1]}]"],
                    ["W2", f"[[{W2[0,0]}, {W2[0,1]}], [{W2[1,0]}, {W2[1,1]}]]"],
                    ["W3", f"[{W3[0]}, {W3[1]}]"],
                ]},
                {"type":"math","content": r"X \to [W_1] \to h_A \to [W_2] \to h_B \to [W_3] \to \hat{y}"},
                {"type":"highlight","content":f"Objetivo: dado X = {X}, a rede deve aprender a produzir ŷ ≈ {Y}.","variant":"teal"},
            ]
        })

        # ── Passo 1: Sigmoide ────────────────────────────────────────────
        steps.append({
            "title": "Função de Ativação: Sigmoide",
            "sections": [
                {"type":"text","content":"A sigmoide mapeia qualquer real para (0, 1), introduzindo não-linearidade. Sua derivada é usada no backpropagation para calcular gradientes."},
                {"type":"math","content": r"\sigma(z) = \frac{1}{1 + e^{-z}}"},
                {"type":"math","content": r"\sigma'(z) = \sigma(z) \cdot \bigl(1 - \sigma(z)\bigr)"},
                {"type":"img","content": plot_sigmoid()},
                {"type":"text","content":f"Valores de referência:  σ(0) = {sigmoid(0):.4f}  ·  σ(2) = {sigmoid(2):.4f}  ·  σ(−2) = {sigmoid(-2):.4f}"},
            ]
        })

        # ── Passo 2: Forward Pass Época 1 ────────────────────────────────
        zA=X*W1; hA=sigmoid(zA)
        zB=np.dot(hA,W2); hB=sigmoid(zB)
        zY=np.dot(hB,W3); yp=sigmoid(zY)
        err1=0.5*(Y-yp)**2

        steps.append({
            "title": "Forward Pass — Época 1",
            "sections": [
                {"type":"text","content":"A informação flui da esquerda para a direita. Cada neurônio calcula uma combinação linear dos valores anteriores e aplica a sigmoide."},
                {"type":"img","content": plot_forward(X,hA,hB,yp,W1,W2,W3,"Época 1 — Forward Pass  (setas →, pesos nas conexões, ativações nos nós)")},
                {"type":"subtitle","content":"Camada Oculta A"},
                {"type":"math","content": r"z_A = X \cdot W_1 = " + f"{X} \\cdot [{W1[0]},\\ {W1[1]}] = [{zA[0]:.4f},\\ {zA[1]:.4f}]"},
                {"type":"math","content": r"h_A = \sigma(z_A) = [" + f"{hA[0]:.4f},\\ {hA[1]:.4f}" + r"]"},
                {"type":"subtitle","content":"Camada Oculta B"},
                {"type":"math","content": r"z_B = h_A \cdot W_2 = [" + f"{hA[0]:.4f},\\ {hA[1]:.4f}" + r"] \cdot \begin{bmatrix}" + f"{W2[0,0]} & {W2[0,1]}" + r"\\" + f"{W2[1,0]} & {W2[1,1]}" + r"\end{bmatrix} = [" + f"{zB[0]:.4f},\\ {zB[1]:.4f}" + r"]"},
                {"type":"math","content": r"h_B = \sigma(z_B) = [" + f"{hB[0]:.4f},\\ {hB[1]:.4f}" + r"]"},
                {"type":"subtitle","content":"Saída"},
                {"type":"math","content": r"z_Y = h_B \cdot W_3 = [" + f"{hB[0]:.4f},\\ {hB[1]:.4f}" + r"] \cdot [" + f"{W3[0]},\\ {W3[1]}" + r"] = " + f"{zY:.4f}"},
                {"type":"math","content": r"\hat{y} = \sigma(z_Y) = " + f"{yp:.4f}"},
                {"type":"highlight","content":f"Predição: ŷ = {yp:.4f}   Alvo: y = {Y}   Erro: E = {err1:.6f}","variant":"orange"},
            ]
        })

        # ── Passo 3: Função de Custo ─────────────────────────────────────
        steps.append({
            "title": "Função de Custo — Época 1",
            "sections": [
                {"type":"text","content":"O Erro Quadrático Médio (MSE) mede a distância entre predição e alvo. O fator ½ cancela o expoente 2 na derivada, simplificando o cálculo do gradiente."},
                {"type":"math","content": r"E = \frac{1}{2}(y - \hat{y})^2"},
                {"type":"math","content": r"E = \frac{1}{2}(" + f"{Y} - {yp:.4f}" + r")^2 = " + f"{err1:.6f}"},
                {"type":"highlight","content":f"A rede prevê {yp:.4f}, mas deveria ser {Y}. Agora calculamos os gradientes para ajustar os pesos.","variant":"orange"},
            ]
        })

        # ── Passo 4: Backpropagation Época 1 ─────────────────────────────
        W1b=W1.copy(); W2b=W2.copy(); W3b=W3.copy()
        dY=(yp-Y)*d_sig(yp)
        dW3=dY*hB
        dhB=dY*W3*d_sig(hB)
        dW2=np.outer(hA,dhB)
        dhA=np.dot(dhB,W2.T)*d_sig(hA)
        dW1=dhA*X
        W3-=lr*dW3; W2-=lr*dW2; W1-=lr*dW1

        steps.append({
            "title": "Backpropagation — Época 1",
            "sections": [
                {"type":"text","content":"A regra da cadeia propaga o sinal de erro da saída para a entrada. Cada neurônio recebe um delta (δ) que indica o quanto contribuiu para o erro."},
                {"type":"img","content": plot_backprop(X,dW1,dW2,dW3,dhA,dhB,dY,"Época 1 — Backpropagation  (setas ←, gradientes nas conexões, δ nos nós)")},
                {"type":"subtitle","content":"Gradientes calculados"},
                {"type":"math","content": r"\delta_Y = (\hat{y} - y) \cdot \sigma'(\hat{y}) = (" + f"{yp:.4f} - {Y}) \\cdot {d_sig(yp):.4f} = {dY:.6f}"},
                {"type":"math","content": r"\nabla W_3 = \delta_Y \cdot h_B = [" + f"{dW3[0]:.4f},\\ {dW3[1]:.4f}" + r"]"},
                {"type":"math","content": r"\delta_{hB} = \delta_Y \cdot W_3 \cdot \sigma'(h_B) = [" + f"{dhB[0]:.6f},\\ {dhB[1]:.6f}" + r"]"},
                {"type":"math","content": r"\nabla W_2 = \begin{bmatrix}" + f"{dW2[0,0]:.4f} & {dW2[0,1]:.4f}" + r"\\" + f"{dW2[1,0]:.4f} & {dW2[1,1]:.4f}" + r"\end{bmatrix}"},
                {"type":"math","content": r"\nabla W_1 = \delta_{hA} \cdot X = [" + f"{dW1[0]:.4f},\\ {dW1[1]:.4f}" + r"]"},
                {"type":"subtitle","content":f"Atualização dos pesos  (W ← W − {lr} · ∇W)"},
                {"type":"math","content": r"W_3:\ [" + f"{W3b[0]:.4f},\\ {W3b[1]:.4f}" + r"] \;\longrightarrow\; [" + f"{W3[0]:.4f},\\ {W3[1]:.4f}" + r"]"},
                {"type":"math","content": r"W_1:\ [" + f"{W1b[0]:.4f},\\ {W1b[1]:.4f}" + r"] \;\longrightarrow\; [" + f"{W1[0]:.4f},\\ {W1[1]:.4f}" + r"]"},
            ]
        })

        # ── Passo 5: Época 2 ─────────────────────────────────────────────
        zA2=X*W1; hA2=sigmoid(zA2)
        zB2=np.dot(hA2,W2); hB2=sigmoid(zB2)
        zY2=np.dot(hB2,W3); yp2=sigmoid(zY2)
        err2=0.5*(Y-yp2)**2

        W1b2=W1.copy(); W2b2=W2.copy(); W3b2=W3.copy()
        dY2=(yp2-Y)*d_sig(yp2)
        dW3_2=dY2*hB2; dhB2=dY2*W3*d_sig(hB2)
        dW2_2=np.outer(hA2,dhB2)
        dhA2=np.dot(dhB2,W2.T)*d_sig(hA2); dW1_2=dhA2*X
        W3-=lr*dW3_2; W2-=lr*dW2_2; W1-=lr*dW1_2

        steps.append({
            "title": "Época 2 — Ciclo Completo",
            "sections": [
                {"type":"text","content":"Repetimos o mesmo ciclo com os pesos já atualizados. O ciclo é sempre: Forward → Erro → Backpropagation → Atualização."},
                {"type":"subtitle","content":"Forward Pass"},
                {"type":"img","content": plot_forward(X,hA2,hB2,yp2,W1,W2,W3,"Época 2 — Forward Pass com os pesos atualizados")},
                {"type":"math","content": r"h_A = [" + f"{hA2[0]:.4f},\\ {hA2[1]:.4f}" + r"] \qquad h_B = [" + f"{hB2[0]:.4f},\\ {hB2[1]:.4f}" + r"]"},
                {"type":"math","content": r"\hat{y} = " + f"{yp2:.4f}" + r"\qquad E = " + f"{err2:.6f}" + r"\quad (\text{época 1: }" + f"{err1:.6f}" + r")"},
                {"type":"highlight","content":f"Melhora: {err1:.6f} → {err2:.6f}  ({(1-err2/err1)*100:.1f}% de redução do erro)","variant":"teal"},
                {"type":"subtitle","content":"Backpropagation"},
                {"type":"img","content": plot_backprop(X,dW1_2,dW2_2,dW3_2,dhA2,dhB2,dY2,"Época 2 — Backpropagation")},
                {"type":"math","content": r"\delta_Y = " + f"{dY2:.6f}" + r"\quad (\text{época 1: }" + f"{dY:.6f}" + r")"},
                {"type":"math","content": r"\nabla W_3 = [" + f"{dW3_2[0]:.4f},\\ {dW3_2[1]:.4f}" + r"] \qquad \nabla W_1 = [" + f"{dW1_2[0]:.4f},\\ {dW1_2[1]:.4f}" + r"]"},
            ]
        })

        # ── Passo 6: Treinamento completo ────────────────────────────────
        hist_e=[err1,err2]; hist_p=[yp,yp2]
        for ep in range(3, epochs+1):
            zA_=X*W1; hA_=sigmoid(zA_)
            zB_=np.dot(hA_,W2); hB_=sigmoid(zB_)
            zY_=np.dot(hB_,W3); yp_=sigmoid(zY_)
            e_=0.5*(Y-yp_)**2
            hist_e.append(e_); hist_p.append(yp_)
            dY_=(yp_-Y)*d_sig(yp_)
            dW3_=dY_*hB_; dhB_=dY_*W3*d_sig(hB_)
            dW2_=np.outer(hA_,dhB_)
            dhA_=np.dot(dhB_,W2.T)*d_sig(hA_); dW1_=dhA_*X
            W3-=lr*dW3_; W2-=lr*dW2_; W1-=lr*dW1_

        marks=[0,9,epochs//4-1,epochs//2-1,epochs-1]
        rows=[]
        for m in sorted(set([max(0,x) for x in marks])):
            rows.append([str(m+1), f"{hist_p[m]:.4f}", f"{hist_e[m]:.6f}"])

        steps.append({
            "title": f"Treinamento Completo — {epochs} Épocas",
            "sections": [
                {"type":"img","content": plot_learning_curve(hist_e,hist_p,Y,lr,X,epochs)},
                {"type":"table","headers":["Época","Predição ŷ","Erro (MSE)"],"rows":rows},
                {"type":"highlight","content":f"Redução total: {hist_e[0]:.6f} → {hist_e[-1]:.6f}  ({(1-hist_e[-1]/hist_e[0])*100:.1f}% de melhora)","variant":"teal"},
            ]
        })

# ── Passo 7: Estado final + Taxonomia ────────────────────────────
        zA_f=X*W1; hA_f=sigmoid(zA_f)
        zB_f=np.dot(hA_f,W2); hB_f=sigmoid(zB_f)
        zY_f=np.dot(hB_f,W3); yp_f=sigmoid(zY_f)

        steps.append({
            "title": f"Estado Final & Taxonomia",
            "sections": [
                {"type":"subtitle","content":"Resumo das Configurações Iniciais"},
                {"type":"table","headers":["Parâmetro","Valor Definido"],"rows":[
                    ["Entrada X", str(X)], ["Alvo y", str(Y)],
                    ["Taxa de aprendizagem (lr)", str(lr)], ["Épocas", str(epochs)],
                    ["Pesos Iniciais W1", f"[{w1_0}, {w1_1}]"],
                    ["Pesos Iniciais W2", f"[[{w2_00}, {w2_01}], [{w2_10}, {w2_11}]]"],
                    ["Pesos Iniciais W3", f"[{w3_0}, {w3_1}]"],
                ]},
                {"type":"subtitle","content":"Resultado Final do Treinamento"},
                {"type":"img","content": plot_forward(X,hA_f,hB_f,yp_f,W1,W2,W3,f"Época {epochs} — Predição final: {yp_f:.4f}  |  Alvo: {Y}")},
                {"type":"math","content": r"\hat{y}_{final} = " + f"{yp_f:.6f}" + r"\quad \approx \quad y = " + f"{Y}"},
                {"type":"math","content": r"E_{final} = " + f"{hist_e[-1]:.8f}"},
                {"type":"highlight","content":f"Predição:  {hist_p[0]:.4f}  (início)  →  {yp_f:.4f}  (final)   alvo = {Y}","variant":"teal"},
                {"type":"subtitle","content":"Onde esta rede se encaixa? Taxonomia das Redes Neurais"},
                {"type":"img","content": plot_taxonomy()},
                {"type":"text","content":"MLP: conexões densas, dados tabulares. CNN: filtros compartilhados, dados espaciais (imagens). RNN: estado oculto com memória, dados sequenciais (texto, séries temporais)."},
            ]
        })

        return json.dumps({"ok":True,"steps":steps})
    except Exception as e:
        import traceback
        return json.dumps({"ok":False,"error":str(e),"tb":traceback.format_exc()})
