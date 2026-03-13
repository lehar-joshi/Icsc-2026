"""
Mean-Field Ising Model — Full Version
======================================
Tokens are ±1 spin configurations projected via W_Q, W_K into query/key space.
J_ij = qi.kj / sqrt(d_k) is a unique coupling per (i,j) pair.
Self-consistent iteration per (i,j) pair gives converged equilibrium mean spin.
Attention emerges from the partition function — not applied.

EXPLICIT ASSUMPTIONS
--------------------
A1. mu = 1  (magnetic moment, explicit)
A2. B_ext = 0  (query absorbed into J_ij)
A3. Self-consistent iteration per (i,j): m <- tanh(beta*mu*q*J_ij*m)
    until |delta_m| < 1e-6. Seeded at sign(J_ij)*0.5.
A4. q = 5 (coordination number). All-to-all is a CONSEQUENCE of <Sj>,
    not an imposed assumption.
A5. W_Q != W_K (asymmetric random projections). J_ij != J_ji in general.
    Spin identity lives in X (+-1). Q, K are continuous after projection.
A6. Two distinct probability objects:
    P_spin_ij = sigmoid(2*h_ij)  where h = beta*mu*q*J*m
        Single-spin Boltzmann prob P(Sj=+1 | query i).
        Normalised over spin states {+1,-1} for each j.
        Use for: tracking microscopic spin state under embedding swap.
    P_attn_ij = exp(h_ij) / sum_j exp(h_ij)
        Emergent attention weight. Normalised over keys j.
        Fell out of Z_i — not applied, derived.
A7. Converged <m_j>_i plays role of value Vj. No separate V matrix.
    <M_i> = sum_j P_attn_ij * m_ij
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

np.random.seed(42)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['font.size']        = 10

# ── Constants ──────────────────────────────────────────────────────────────────
mu       = 1.0
beta     = 1.0
q        = 5
tol      = 1e-6
max_iter = 10000
d_model  = 8
d_k      = 8     # head dimension — 1/sqrt(d_k) maps to beta in Ising picture

print("=" * 65)
print("ASSUMPTIONS")
print("=" * 65)
print(f"  mu      = {mu}    (A1: explicit)")
print(f"  B_ext   = 0      (A2: query absorbed into J_ij)")
print(f"  beta    = {beta}")
print(f"  q       = {q}      (A4: coordination number)")
print(f"  tol     = {tol}  (A3: convergence per pair)")
print(f"  d_model = {d_model}, d_k = {d_k}  (A5: head dimension)")
print()
print("  All-to-all coupling is a CONSEQUENCE of mean-field <Sj>,")
print("  not an imposed assumption. (A4)")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Token spin configurations
# ═══════════════════════════════════════════════════════════════════════════════

token_spins = {
    'The' : np.array([ 1,  1, -1,  1, -1,  1,  1, -1], dtype=float),
    'cat' : np.array([ 1,  1, -1,  1, -1, -1,  1,  1], dtype=float),
    'sat' : np.array([-1,  1,  1, -1,  1, -1,  1,  1], dtype=float),
}
tokens_small = ['The', 'cat', 'sat']
X_small = np.array([token_spins[t] for t in tokens_small])

print("=" * 65)
print("SECTION 1 — Token spin configurations")
print("=" * 65)
for t in tokens_small:
    ups   = int((token_spins[t] ==  1).sum())
    downs = int((token_spins[t] == -1).sum())
    print(f"  '{t:4s}' = {token_spins[t].astype(int).tolist()}  ({ups}up {downs}down)")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Projection and coupling
# ═══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5


def compute_QK(X):
    """A5: continuous after projection. Spin identity lives in X, not Q/K."""
    return X @ W_Q, X @ W_K


def compute_J(Q, K):
    """J_ij = qi.kj / sqrt(d_k). B_ext=0 (A2), J_ij is full effective field."""
    return Q @ K.T / np.sqrt(d_k)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Self-consistent mean-field iteration (A3)
# ═══════════════════════════════════════════════════════════════════════════════

def self_consistent_m(J, beta, mu, q, tol, max_iter):
    """
    Solve  m_ij <- tanh(beta*mu*q*J_ij*m_ij)  per (i,j) pair independently.

    Seeded at sign(J_ij)*0.5 to land on the correct side of bifurcation.
    n_iters initialised to max_iter — unconverged pairs show max_iter.

    Returns
    -------
    m       : (N_q, N_k)  converged equilibrium mean spin
    n_iters : (N_q, N_k)  iterations to convergence per pair
    """
    # seed on correct side of bifurcation — critical for convergence
    m            = np.sign(J) * 0.5
    m[m == 0]    = 0.5
    n_iters      = np.full_like(J, max_iter, dtype=int)
    scale        = beta * mu * q

    for n in range(max_iter):
        m_new   = np.tanh(scale * J * m)
        delta   = np.abs(m_new - m)
        # record FIRST iteration where each pair converged
        newly   = (delta < tol) & (n_iters == max_iter)
        n_iters[newly] = n
        m       = m_new
        if delta.max() < tol:
            break

    return m, n_iters


def mean_field(J, beta, mu, q, tol, max_iter):
    """
    All observables from mean-field Ising partition function.

    h_ij = beta*mu*q*J_ij*m_ij  — effective field at convergence (m inside!)

    Returns
    -------
    m       (N_q, N_k)  converged <m_j>_i      [= Vj, A7]
    Z       (N_q,)      Z_i = prod_j 2*cosh(h_ij)
    F       (N_q,)      F_i = -(1/beta)*ln(Z_i)
    P_spin  (N_q, N_k)  sigmoid(2*h_ij)         [A6-obj1, normalised over {+1,-1}]
    P_attn  (N_q, N_k)  exp(h_ij)/sum_j(h_ij)  [A6-obj2, normalised over j]
    E       (N_q, N_k)  E_ij = -mu*q*J_ij*m_ij
    M       (N_q,)      <M_i> = sum_j P_attn_ij * m_ij
    n_iters (N_q, N_k)  iterations to convergence
    """
    m, n_iters = self_consistent_m(J, beta, mu, q, tol, max_iter)

    # effective field at convergence — m MUST be here (self-consistent feedback)
    h = beta * mu * q * J * m

    # partition function — product over independent spins (mean-field decoupling)
    log_Z = np.sum(np.log(2.0 * np.cosh(h)), axis=1)
    Z     = np.exp(log_Z)
    F     = -(1.0 / beta) * log_Z

    # A6-obj1: single-spin Boltzmann probability
    # P(Sj=+1|i) = exp(h) / (exp(h) + exp(-h)) = sigmoid(2h)
    # normalised over spin states {+1,-1} — do NOT sum over j
    P_spin = 1.0 / (1.0 + np.exp(-2.0 * h))

    # A6-obj2: emergent attention weight
    # normalised over keys j — this is what fell out of Z
    h_stable = h - h.max(axis=1, keepdims=True)
    P_attn   = np.exp(h_stable) / np.exp(h_stable).sum(axis=1, keepdims=True)

    # pair energy at equilibrium
    E = -mu * q * J * m

    # magnetisation: proper expectation over keys weighted by P_attn (A7)
    M = (P_attn * m).sum(axis=1)

    return m, Z, F, P_spin, P_attn, E, M, n_iters


Q_s, K_s = compute_QK(X_small)
J_s      = compute_J(Q_s, K_s)
m, Z, F, P_spin, P_attn, E, M, n_iters = mean_field(
    J_s, beta, mu, q, tol, max_iter)

print("=" * 65)
print(f"SECTION 3 — Mean-field observables  (beta={beta}, mu={mu}, q={q})")
print("=" * 65)
for i, t in enumerate(tokens_small):
    print(f"  Query '{t}'")
    print(f"    <m_j>  = {np.round(m[i], 6)}   (converged, A3)")
    print(f"    iters  = {n_iters[i]}")
    print(f"    E_ij   = {np.round(E[i], 4)}")
    print(f"    P_spin = {np.round(P_spin[i], 4)}   (single-spin Boltzmann, A6-obj1)")
    print(f"    P_attn = {np.round(P_attn[i], 4)}   (emergent attention,    A6-obj2)")
    print(f"    Z_i    = {Z[i]:.4f}   F_i = {F[i]:.4f}   <M_i> = {M[i]:.6f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Spin flip perturbation
# ═══════════════════════════════════════════════════════════════════════════════

token_spins_p = {k: v.copy() for k, v in token_spins.items()}
flip_token    = 'cat'
flip_spin     = 2
token_spins_p[flip_token][flip_spin] *= -1

print("=" * 65)
print(f"SECTION 4 — Spin flip: '{flip_token}' spin {flip_spin}  "
      f"{int(token_spins[flip_token][flip_spin]):+d} -> "
      f"{int(token_spins_p[flip_token][flip_spin]):+d}")
print("=" * 65)

X_p = np.array([token_spins_p[t] for t in tokens_small])
Q_p, K_p = compute_QK(X_p)
J_p      = compute_J(Q_p, K_p)
m_p, Z_p, F_p, P_spin_p, P_attn_p, E_p, M_p, _ = mean_field(
    J_p, beta, mu, q, tol, max_iter)

dP_spin = P_spin_p  - P_spin
dP_attn = P_attn_p  - P_attn
max_idx = np.unravel_index(np.abs(dP_attn).argmax(), dP_attn.shape)

print(f"  Most changed P_attn: ({tokens_small[max_idx[0]]} -> {tokens_small[max_idx[1]]})  "
      f"delta = {dP_attn[max_idx]:+.6f}")
print(f"  <M> before : {np.round(M,   6)}")
print(f"  <M> after  : {np.round(M_p, 6)}")
print(f"  delta<M>   : {np.round(M_p - M, 6)}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Large sentence (12 tokens, 4 lattice groups)
# ═══════════════════════════════════════════════════════════════════════════════

sentence = [
    'the',    'old',    'black',
    'cat',    'sat',    'quietly',
    'on',     'the2',   'warm',
    'wooden', 'mat',    'today',
]
lattice_labels = [
    'Lattice 1\n(noun mod)',
    'Lattice 2\n(subj+verb)',
    'Lattice 3\n(prep)',
    'Lattice 4\n(object)',
]
lattice_size = 3
N_large      = len(sentence)

np.random.seed(7)
X_large = np.array([
    np.random.choice([-1.0, 1.0], size=d_model) for _ in sentence
])
Q_l, K_l = compute_QK(X_large)
J_l      = compute_J(Q_l, K_l)
m_l, Z_l, F_l, P_spin_l, P_attn_l, E_l, M_l, _ = mean_field(
    J_l, beta, mu, q, tol, max_iter)

# flip spin 2 of 'cat' (index 3)
X_large_p        = X_large.copy()
X_large_p[3, 2] *= -1
Q_lp, K_lp       = compute_QK(X_large_p)
J_lp             = compute_J(Q_lp, K_lp)
m_lp, Z_lp, F_lp, P_spin_lp, P_attn_lp, E_lp, M_lp, _ = mean_field(
    J_lp, beta, mu, q, tol, max_iter)

dP_attn_l = P_attn_lp - P_attn_l
max_idx_l = np.unravel_index(np.abs(dP_attn_l).argmax(), dP_attn_l.shape)

print("=" * 65)
print("SECTION 5 — Large sentence")
print("=" * 65)
print(f"  Most changed cell after flip: "
      f"({sentence[max_idx_l[0]]} -> {sentence[max_idx_l[1]]})  "
      f"delta = {dP_attn_l[max_idx_l]:+.6f}")
print()

N_range = np.arange(3, 257)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Token spin configs ───────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 3))
sns.heatmap(X_small, annot=True, fmt='.0f', cmap='RdBu', center=0,
            xticklabels=[f's{i}' for i in range(d_model)],
            yticklabels=tokens_small,
            linewidths=1, ax=ax, vmin=-1, vmax=1,
            cbar_kws={'label': 'Spin (+-1)'})
ax.set_title('Token Spin Configurations  (+-1 per spin, illustrative)\n'
             'Spin identity lives in X — lost after W_Q/W_K projection (A5)',
             fontsize=11)
ax.set_xlabel('Spin index')
ax.set_ylabel('Token')
plt.tight_layout()
plt.show()


# ── Figure 2: Converged <m>, energy, P_spin, P_attn ───────────────────────────
fig2, axes = plt.subplots(1, 4, figsize=(20, 4.5))

vmax_m = np.abs(m).max()
sns.heatmap(m, annot=True, fmt='.4f', cmap='RdBu', center=0,
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[0], vmin=-vmax_m, vmax=vmax_m)
axes[0].set_title('Converged $\\langle m_j \\rangle_i$\n'
                  '$m \\leftarrow \\tanh(\\beta\\mu q J_{ij} m)$\n'
                  'Equilibrium mean spin (A3)  =  $V_j$ (A7)', fontsize=9)
axes[0].set_xlabel('Key $k_j$'); axes[0].set_ylabel('Query $q_i$')

vmax_e = np.abs(E).max()
sns.heatmap(E, annot=True, fmt='.3f', cmap='YlOrRd_r',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[1], vmin=-vmax_e, vmax=vmax_e)
axes[1].set_title('Pair Energy\n'
                  '$E_{ij} = -\\mu q J_{ij} \\langle m_j \\rangle_i$\n'
                  'Yellow = low energy = stable = highest attention', fontsize=9)
axes[1].set_xlabel('Key $k_j$'); axes[1].set_ylabel('Query $q_i$')

sns.heatmap(P_spin, annot=True, fmt='.4f', cmap='Greens',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[2], vmin=0, vmax=1)
axes[2].set_title('$P_{spin}$  (A6-obj1)\n'
                  '$\\sigma(2h_{ij})$,  $h=\\beta\\mu q J m$\n'
                  'Single-spin Boltzmann  P$(S_j=+1|i)$\n'
                  'Normalised over $\\{+1,-1\\}$ — do not sum over $j$', fontsize=9)
axes[2].set_xlabel('Key $k_j$'); axes[2].set_ylabel('Query $q_i$')

sns.heatmap(P_attn, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[3], vmin=0, vmax=1)
axes[3].set_title('$P_{attn}$  (A6-obj2)\n'
                  '$e^{h_{ij}} / \\sum_j e^{h_{ij}}$\n'
                  'Emergent attention — fell out of $Z_i$\n'
                  'Normalised over keys $j$', fontsize=9)
axes[3].set_xlabel('Key $k_j$'); axes[3].set_ylabel('Query $q_i$')

plt.suptitle(f'Mean-Field Observables  (beta={beta}, mu={mu}, q={q}, tol={tol})\n'
             'h = beta*mu*q*J*m  (converged m in exponent — self-consistent feedback)',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 3: Energy vs P_attn anticorrelation ────────────────────────────────
fig3, ax = plt.subplots(figsize=(6, 5))
colors_s = plt.cm.tab10(np.arange(len(tokens_small)))
for i, t in enumerate(tokens_small):
    ax.scatter(E[i], P_attn[i], s=140, color=colors_s[i],
               label=f'Query "{t}"', zorder=3)
    for j, tj in enumerate(tokens_small):
        ax.annotate(f'→{tj}', (E[i, j], P_attn[i, j]),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)

flat_idx = E.argmin()
qi, kj   = np.unravel_index(flat_idx, E.shape)
ax.annotate(f'Most stable\n({tokens_small[qi]}→{tokens_small[kj]})',
            xy=(E[qi, kj], P_attn[qi, kj]),
            xytext=(E[qi, kj] + 0.05, P_attn[qi, kj] - 0.12),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8, color='red')

ax.set_xlabel('Pair Energy  $E_{ij}$')
ax.set_ylabel('Emergent Attention  $P_{attn,ij}$')
ax.set_title('Low Energy → High Attention\nStability drives selection — derived not assumed',
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── Figure 4: Spin flip — P_spin and P_attn before/after/diff ────────────────
fig4, axes = plt.subplots(2, 3, figsize=(15, 8))

r, c = max_idx
for row, (P_before, P_after, label, cmap) in enumerate([
    (P_spin,  P_spin_p,  'P_spin  (single-spin Boltzmann, A6-obj1)', 'Greens'),
    (P_attn,  P_attn_p,  'P_attn  (emergent attention,    A6-obj2)', 'Blues'),
]):
    dP     = P_after - P_before
    vmax_d = np.abs(dP).max()

    sns.heatmap(P_before, annot=True, fmt='.4f', cmap=cmap,
                xticklabels=tokens_small, yticklabels=tokens_small,
                linewidths=0.5, ax=axes[row, 0], vmin=0, vmax=1)
    axes[row, 0].set_title(f'{label}\nBefore flip', fontsize=9)
    axes[row, 0].set_xlabel('Key'); axes[row, 0].set_ylabel('Query')

    sns.heatmap(P_after, annot=True, fmt='.4f', cmap=cmap,
                xticklabels=tokens_small, yticklabels=tokens_small,
                linewidths=0.5, ax=axes[row, 1], vmin=0, vmax=1)
    axes[row, 1].set_title(f'{label}\nAfter flip: "{flip_token}" spin {flip_spin} '
                           f'{int(token_spins[flip_token][flip_spin]):+d}->'
                           f'{int(token_spins_p[flip_token][flip_spin]):+d}', fontsize=9)
    axes[row, 1].set_xlabel('Key'); axes[row, 1].set_ylabel('Query')
    # circle most changed cell
    if row == 1:
        axes[row, 1].add_patch(mpatches.Circle(
            (c + 0.5, r + 0.5), 0.45,
            fill=False, edgecolor='red', lw=3, zorder=5))

    sns.heatmap(dP, annot=True, fmt='+.4f', cmap='coolwarm', center=0,
                xticklabels=tokens_small, yticklabels=tokens_small,
                linewidths=0.5, ax=axes[row, 2], vmin=-vmax_d, vmax=vmax_d)
    axes[row, 2].set_title('Delta  (After - Before)\n'
                           + ('Which spins changed state?' if row == 0
                              else 'Which attention weights shifted?'), fontsize=9)
    axes[row, 2].set_xlabel('Key'); axes[row, 2].set_ylabel('Query')
    if row == 1:
        axes[row, 2].add_patch(mpatches.Circle(
            (c + 0.5, r + 0.5), 0.45,
            fill=False, edgecolor='red', lw=3, zorder=5))

plt.suptitle(f'Spin Flip: "{flip_token}" spin {flip_spin}\n'
             f'Row 1: P_spin tracks microscopic spin state  |  '
             f'Row 2: P_attn tracks attention shift\n'
             f'Most changed: ({tokens_small[r]} -> {tokens_small[c]})  '
             f'dP = {dP_attn[max_idx]:+.6f}',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 5: Mean magnetisation before/after ─────────────────────────────────
fig5, ax = plt.subplots(figsize=(8, 4))
x     = np.arange(len(tokens_small))
width = 0.35
ax.bar(x - width/2, M,   width, color='steelblue', alpha=0.85, label='Before flip')
ax.bar(x + width/2, M_p, width, color='coral',     alpha=0.85, label='After flip')

for i in range(len(tokens_small)):
    delta = M_p[i] - M[i]
    ax.annotate(f'Δ={delta:+.4f}',
                xy=(x[i] + width / 2, M_p[i]),
                xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=8,
                color='red' if abs(delta) > 1e-4 else 'gray')

ax.set_xticks(x)
ax.set_xticklabels([f'Query "{t}"' for t in tokens_small])
ax.set_ylabel('$\\langle M_i \\rangle = \\sum_j P_{attn,ij} \\langle m_j \\rangle_i$')
ax.set_title('Mean Magnetisation per Dipole — Before vs After Spin Flip\n'
             'Converged $\\langle m_j \\rangle_i$ is $V_j$ (A7)\n'
             'Large Δ = that query felt the perturbation', fontsize=10)
ax.legend()
ax.axhline(0, color='black', lw=0.8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# ── Figure 6: Large sentence P_attn before/after with circles ─────────────────
fig6, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, P_plot, title in zip(
    axes,
    [P_attn_l, P_attn_lp],
    ['$P_{attn}$ Before flip  (12-token sentence)',
     '$P_{attn}$ After flip — "cat" spin 2 flipped']
):
    sns.heatmap(P_plot, cmap='Blues', ax=ax,
                xticklabels=sentence, yticklabels=sentence,
                linewidths=0.2, vmin=0, vmax=P_attn_l.max())
    for boundary in [3, 6, 9]:
        ax.axhline(boundary, color='red', lw=2)
        ax.axvline(boundary, color='red', lw=2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Key $k_j$')
    ax.set_ylabel('Query $q_i$')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# circle most changed cell
r_l, c_l = max_idx_l
axes[1].add_patch(mpatches.Circle(
    (c_l + 0.5, r_l + 0.5), 0.45,
    fill=False, edgecolor='orange', lw=3, zorder=5))
axes[1].text(c_l + 0.5, r_l - 0.4,
             f'most changed\n({sentence[r_l]}→{sentence[c_l]})',
             ha='center', va='bottom', fontsize=8,
             color='orange', fontweight='bold')

for g in range(4):
    mid = g * lattice_size + lattice_size / 2
    axes[0].text(1.01, 1.0 - mid / N_large, lattice_labels[g],
                 ha='left', va='center', fontsize=7, color='red',
                 fontweight='bold', transform=axes[0].transAxes)

plt.suptitle('Large Sentence (12 tokens, 4 lattice groups)\n'
             'Red lines = lattice boundaries  |  '
             'Orange circle = cell that changed most after spin flip',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 7: O(N^2) cost ─────────────────────────────────────────────────────
fig7, axes = plt.subplots(1, 2, figsize=(13, 5))
ops = N_range ** 2

axes[0].plot(N_range, ops, color='navy', lw=2)
axes[0].fill_between(N_range, ops, alpha=0.12, color='navy')
for n, label in [(3, '"The cat sat"'), (12, 'Our sentence'),
                 (64, '64 tokens'),    (128, '128 tokens')]:
    axes[0].scatter(n, n**2, s=80, color='red', zorder=5)
    axes[0].annotate(f'{label}\n{n**2:,} ops',
                     xy=(n, n**2), xytext=(n + 3, n**2 * 0.75),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1),
                     fontsize=8, color='red')
axes[0].set_xlabel('Number of tokens N')
axes[0].set_ylabel('Pairwise $J_{ij}$ computations ($N^2$)')
axes[0].set_title('$O(N^2)$ Cost of the Full Partition Function\n'
                  'Each (i,j) pair converges independently\n'
                  'All-to-all emerges from $\\langle S_j \\rangle$', fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].loglog(N_range, ops,     color='darkred',   lw=2, label='$N^2$ (pairwise)')
axes[1].loglog(N_range, N_range, color='steelblue', lw=2, ls='--', label='$N$ (linear)')
axes[1].set_xlabel('N  (log)')
axes[1].set_ylabel('Operations  (log)')
axes[1].set_title('Log-Log: $N^2$ vs $N$\n'
                  'Longer sentence = larger spin lattice = more expensive', fontsize=10)
axes[1].legend()
axes[1].grid(True, alpha=0.3, which='both')
plt.suptitle('Why Large Spin Systems Are Expensive\n'
             '$O(N^2)$ pairwise couplings — consequence of mean-field all-to-all',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Mapping table ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MAPPING TABLE — Mean-Field Ising <-> Attention")
print("=" * 65)
rows = [
    ("Energy  E_ij = -mu*q*J_ij*m_ij",             "Stability of query-key pair"),
    ("Effective field  h = beta*mu*q*J*m",          "Scaled similarity with feedback"),
    ("Inv. temp  beta = 1/sqrt(d_k)",               "Scaling factor"),
    ("Coordination number q=5",                      "Each spin sees 5 neighbours avg"),
    ("Converged m_ij  (self-consistent, A3)",        "Value Vj  (A7)"),
    ("Z_i = prod_j 2*cosh(h_ij)",                   "Softmax denominator"),
    ("P_spin = sigmoid(2*h_ij)  [A6-obj1]",         "Single-spin Boltzmann (microscopic)"),
    ("P_attn = exp(h_ij)/Z_i    [A6-obj2, emerges]","Attention weight (not applied)"),
    ("<M_i> = sum_j P_attn_ij * m_ij",             "Attention output"),
    ("Spins / states",                               "Tokens"),
    ("J_ij = qi.kj / sqrt(d_k)",                    "Similarity score"),
    ("beta->inf  (T->0, ordered)",                   "Sharp / peaked attention"),
    ("beta->0   (T->inf, disordered)",               "Uniform / flat attention"),
    ("F_i = -(1/beta)*ln(Z_i)",                     "Log partition function"),
    ("All-to-all from <Sj>  (A4)",                  "All-to-all attention (not designed)"),
]
print(f"  {'Mean-Field Ising':<48}  Attention")
print(f"  {'-'*48}  {'-'*30}")
for left, right in rows:
    print(f"  {left:<48}  {right}")
print()