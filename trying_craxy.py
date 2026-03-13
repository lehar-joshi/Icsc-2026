"""
Mean-Field Ising Model — Attention Emerges from Physics
========================================================

EXPLICIT ASSUMPTIONS
--------------------
A1. mu = 1
    Magnetic moment set to 1 explicitly throughout.

A2. B_ext = 0
    Query absorbed into J_ij = qi.kj / sqrt(d_k). Not modelling
    ferromagnetic phase conditions — deriving attention.

A3. Self-consistent iteration per (i,j) pair
    Each pair converges independently:
        <m_j>_i <- tanh(beta * q * J_ij * <m_j>_i)
    until |<m>_new - <m>_old| < 1e-6
    Converged <m_j>_i is the true equilibrium magnetisation of key
    spin j under query i's field. This is what makes <m> meaningful.

A4. Coordination number q = 5
    Each spin interacts with 5 neighbours on average (Curie-Weiss).
    All-to-all coupling is a consequence of using <Sj>, not imposed.

A5. W_Q != W_K (random projections kept)
    Asymmetric Q, K projections preserved — J_ij != J_ji in general.
    Spin identity lives in X (+-1). Q, K are continuous after projection.

A6. Two distinct probability objects
    P_spin_ij  = exp(beta*q*J_ij) / 2*cosh(beta*q*J_ij)
        Single-spin Boltzmann prob P(Sj=+1 | query i).
        Use for: tracking microscopic spin state under embedding swap.

    P_attn_ij  = exp(beta*q*J_ij) / sum_j exp(beta*q*J_ij)
        Emergent attention weight — fell out of Z_i, not applied.
        Use for: which key does query i focus on?

A7. <m_j>_i (converged) plays the role of value Vj
    No separate V matrix. Equilibrium mean spin IS the value.
    <M_i> = sum_j P_attn_ij * <m_j>_i
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

np.random.seed(42)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['font.size']        = 10


mu      = 1.0    # A1
beta    = 1.0
q       = 5      # A4: coordination number
tol     = 1e-6   # A3: convergence tolerance
max_iter = 10000
d_model  = 8
d_k      = 8 # is this headsize?

print("=" * 65)
print("ASSUMPTIONS")
print("=" * 65)
print(f"  mu     = {mu}   (A1: explicit)")
print(f"  B_ext  = 0     (A2: query absorbed into J_ij)")
print(f"  beta   = {beta}")
print(f"  q      = {q}   (A4: coordination number)")
print(f"  tol    = {tol}  (A3: self-consistent convergence)")
print(f"  d_model= {d_model}, d_k = {d_k}")
print()


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
    print(f"  '{t:4s}' = {token_spins[t].astype(int).tolist()}  "
          f"({ups} up, {downs} down)")
print()


#random projection and coupling, both are never equal to each other
#is that always the case?

np.random.seed(42)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5


def compute_QK(X):
    # A5: continuous after projection — spin identity lives in X, not Q/K
    return X @ W_Q, X @ W_K


def compute_J(Q, K):
    # J_ij = qi.kj / sqrt(d_k)  (B_ext=0, A2)
    return Q @ K.T / np.sqrt(d_k)



# Per (i,j) pair, solve independently:
#   m <- tanh(beta * mu * q * J_ij * m)
# until |m_new - m_old| < tol
#
# The converged m_ij is the equilibrium magnetisation of key spin j
# under query i's field. Without this, <m> is just tanh of a number
# and carries no physical meaning.

def self_consistent_m(J, beta, mu, q, tol, max_iter):
    """
    m        : (N_q, N_k)  converged equilibrium mean spin per pair
    n_iters  : (N_q, N_k)  iterations taken per pair
    """
    m       = np.full_like(J, 0.5)   # seed away from 0
    n_iters = np.zeros_like(J, dtype=int)
    scale   = beta * q

    for n in range(max_iter):
        m_new = np.tanh(scale * J * m)
        delta = np.abs(m_new - m)
        m     = m_new
        converged = (delta < tol) & (n_iters == 0)
        n_iters[converged] = n
        if delta.max() < tol:
            break

    return m, n_iters


def mean_field(J, beta, mu, q, tol, max_iter):
    """
    All observables from the mean-field Ising partition function.
    <m> is the converged self-consistent equilibrium magnetisation.

    Returns
    -------
    m        (N_q, N_k)  converged <m_j>_i  [plays role of V, A7]
    Z        (N_q,)      Z_i = prod_j 2*cosh(beta*mu*q*J_ij)
    F        (N_q,)      F_i = -(1/beta)*ln(Z_i)
    P_spin   (N_q, N_k)  P(Sj=+1|i) = exp(bmqJ) / 2*cosh(bmqJ)   [A6-obj1]
    P_attn   (N_q, N_k)  exp(bmqJ) / sum_j exp(bmqJ)              [A6-obj2]
    E        (N_q, N_k)  E_ij = -mu*q*J_ij*m_ij
    M        (N_q,)      <M_i> = sum_j P_attn_ij * m_ij
    n_iters  (N_q, N_k)  iterations to convergence per pair
    """
    m, n_iters = self_consistent_m(J, beta, mu, q, tol, max_iter)

    bmqJ = beta * mu * q * J   #where is the spin of particle i?

    log_Z = np.sum(np.log(2.0 * np.cosh(bmqJ)), axis=1)
    Z     = np.exp(log_Z)
    F     = -(1.0 / beta) * log_Z # why do we need to calculate that?

    #single-spin Boltzmann probability associated with spin 1? 
    #shouldn't it be devided by z
    #P_spin = np.exp(bmqJ) / (2.0 * np.cosh(bmqJ))
    # exp(bmqJ) / (exp(bmqJ) + exp(-bmqJ)) = sigmoid(2*bmqJ)
    P_spin = 1.0 / (1.0 + np.exp(-2.0 * bmqJ))

    #emergent attention weight (fell out of Z, not applied)
    # I think its purely normalization
    bmqJ_s  = bmqJ - bmqJ.max(axis=1, keepdims=True)
    exp_bmqJ = np.exp(bmqJ_s)
    #P_attn   = exp_bmqJ / exp_bmqJ.sum(axis=1, keepdims=True) # what is this trying to do?
    #result of attentions logits, each elememt is e^bqmJ
    #divided by sums values acrosss axis=1 rows. 
    #keeping dimension so broadcasting words.
     
    #Pair energy that is converged <m> that is mean spin magneizd per dipole.
    E = -mu * q * J * m

    #<M> is magnetization
    M = (P_spin * m).sum(axis=1)

    return m, Z, F, P_spin, E, M, n_iters


Q_s, K_s = compute_QK(X_small)
J_s      = compute_J(Q_s, K_s)
m, Z, F, P_spin, E, M, n_iters = mean_field(
    J_s, beta, mu, q, tol, max_iter)

print("=" * 65)
print(f"SECTION 3 — Self-consistent mean-field  (beta={beta}, mu={mu}, q={q})")
print("=" * 65)
for i, t in enumerate(tokens_small):
    print(f"  Query '{t}'")
    print(f"    <m_j>  = {np.round(m[i], 6)}   (converged, A3)")
    print(f"    iters  = {n_iters[i]}")
    print(f"    E_ij   = {np.round(E[i], 4)}")
    print(f"    P_spin = {np.round(P_spin[i], 4)}   (single-spin Boltzmann, A6-obj1)")
    #print(f"    P_attn = {np.round(P_attn[i], 4)}   (emergent attention,    A6-obj2)")
    print(f"    Z_i    = {Z[i]:.4f}   F_i = {F[i]:.4f}   <M_i> = {M[i]:.6f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Spin flip perturbation
# Flip one spin in 'cat'. Track BOTH P objects.
#   P_spin  -> which spins changed state microscopically?
#   P_attn  -> which attention weights shifted?
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

X_p                                          = np.array([token_spins_p[t] for t in tokens_small])
Q_p, K_p                                     = compute_QK(X_p)
J_p                                          = compute_J(Q_p, K_p)
m_p, Z_p, F_p, P_spin_p, E_p, M_p, _ = mean_field(
    J_p, beta, mu, q, tol, max_iter)

dP_spin  = P_spin_p - P_spin
#dP_attn  = P_attn_p - P_attn

# Find cell that changed most in P_attn
#max_idx         = np.unravel_index(np.abs(dP_attn).argmax(), dP_attn.shape)
#max_query_token = tokens_small[max_idx[0]]
#max_key_token   = tokens_small[max_idx[1]]

#print(f"  Most changed P_attn cell: ({max_query_token} -> {max_key_token})  ")
#print()
#print(f"  <M> before : {np.round(M,   6)}")
#print(f"  <M> after  : {np.round(M_p, 6)}")
#print(f"  delta<M>   : {np.round(M_p - M, 6)}")
#print()


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
X_large  = np.array([
    np.random.choice([-1.0, 1.0], size=d_model)
    for _ in sentence
])
Q_l, K_l  = compute_QK(X_large)
J_l       = compute_J(Q_l, K_l)
m_l, Z_l, F_l, P_spin_l,E_l, M_l, _ = mean_field(
    J_l, beta, mu, q, tol, max_iter)

# Flip one spin in 'cat' (index 3 in sentence)
X_large_p        = X_large.copy()
X_large_p[3, 2] *= -1   # flip spin 2 of 'cat'
Q_lp, K_lp       = compute_QK(X_large_p)
J_lp             = compute_J(Q_lp, K_lp)
m_lp, Z_lp, F_lp, P_spin_lp, E_lp, M_lp, _ = mean_field(
    J_lp, beta, mu, q, tol, max_iter)

dP_attn_l   = P_spin_lp - P_spin_l
max_idx_l   = np.unravel_index(np.abs(dP_attn_l).argmax(), dP_attn_l.shape)

print("=" * 65)
print("SECTION 5 — Large sentence")
print("=" * 65)
print(f"  Most changed cell after flip: ")
print()

N_range = np.arange(3, 257)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Token spin configs ───────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 3))
spin_matrix = np.array([token_spins[t] for t in tokens_small])
sns.heatmap(spin_matrix, annot=True, fmt='.0f', cmap='RdBu', center=0,
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


# ── Figure 2: Converged <m>, energy, both P objects ───────────────────────────
fig2, axes = plt.subplots(1, 4, figsize=(20, 4.5))

vmax_m = np.abs(m).max()
sns.heatmap(m, annot=True, fmt='.4f', cmap='RdBu', center=0,
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[0], vmin=-vmax_m, vmax=vmax_m)
axes[0].set_title('Converged $\\langle m_j \\rangle_i$\n'
                  '$m \\leftarrow \\tanh(\\beta q J_{ij} m)$\n'
                  'Equilibrium mean spin per pair (A3)\n'
                  'Also $V_j$ — the value (A7)', fontsize=9)
axes[0].set_xlabel('Key $k_j$'); axes[0].set_ylabel('Query $q_i$')

vmax_e = np.abs(E).max()
sns.heatmap(E, annot=True, fmt='.3f', cmap='YlOrRd_r',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[1], vmin=-vmax_e, vmax=vmax_e)
axes[1].set_title('Pair Energy\n'
                  '$E_{ij} = -\\mu q J_{ij} \\langle m_j \\rangle_i$\n'
                  'low energy = stable = highest attention', fontsize=9)
axes[1].set_xlabel('Key $k_j$'); axes[1].set_ylabel('Query $q_i$')

sns.heatmap(P_spin, annot=True, fmt='.4f', cmap='Greens',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[2], vmin=0, vmax=1)
axes[2].set_title('$P_{spin}$  (A6-obj1)\n'
                  '$P(S_j=+1|i) = e^{\\beta\\mu q J} / 2\\cosh(\\beta\\mu q J)$\n'
                  'Single-spin Boltzmann probability\n'
                  'Use for: tracking spin state under embedding swap', fontsize=9)
axes[2].set_xlabel('Key $k_j$'); axes[2].set_ylabel('Query $q_i$')


plt.suptitle(f'Mean-Field Observables  (beta={beta}, mu={mu}, q={q}, '
             f'tol={tol})\nConverged self-consistent <m> feeds all downstream quantities',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 3: Spin flip — P_attn before / after / diff + circle ───────────────
fig3, axes = plt.subplots(1, 3, figsize=(15, 4.5))

vmax_p = 1.0
sns.heatmap(P_spin, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[0], vmin=0, vmax=vmax_p)
axes[0].set_title('$P_{attn}$ Before flip', fontsize=10)
axes[0].set_xlabel('Key $k_j$'); axes[0].set_ylabel('Query $q_i$')

sns.heatmap(P_spin_p, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[1], vmin=0, vmax=vmax_p)
axes[1].set_title(f'$P_{{attn}}$ After flip\n'
                  f'"{flip_token}" spin {flip_spin}: '
                  f'{int(token_spins[flip_token][flip_spin]):+d} -> '
                  f'{int(token_spins_p[flip_token][flip_spin]):+d}', fontsize=10)
axes[1].set_xlabel('Key $k_j$'); axes[1].set_ylabel('Query $q_i$')

# Circle the most changed cell on after heatmap
r, c = max_idx
axes[1].add_patch(mpatches.Circle(
    (c + 0.5, r + 0.5), 0.45,
    fill=False, edgecolor='red', lw=3, zorder=5))
axes[1].text(c + 0.5, r - 0.15, 'most\nchanged',
             ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

vmax_d = np.abs(dP_attn).max()
sns.heatmap(dP_attn, annot=True, fmt='+.4f', cmap='coolwarm', center=0,
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[2], vmin=-vmax_d, vmax=vmax_d)
axes[2].set_title('$\\Delta P_{attn}$ = After - Before\n'
                  'Red = attention increased\n'
                  'Blue = attention decreased', fontsize=10)
axes[2].set_xlabel('Key $k_j$'); axes[2].set_ylabel('Query $q_i$')

# Circle most changed cell on diff heatmap too
axes[2].add_patch(mpatches.Circle(
    (c + 0.5, r + 0.5), 0.45,
    fill=False, edgecolor='red', lw=3, zorder=5))

plt.suptitle(f'Spin Flip: "{flip_token}" spin {flip_spin}  →  '
             f'Most changed: ({max_query_token} → {max_key_token})  '
             f'ΔP = {dP_attn[max_idx]:+.6f}',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 4: Mean magnetisation before/after flip ────────────────────────────
fig4, ax = plt.subplots(figsize=(8, 4))
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
             '$\\langle m_j \\rangle_i$ is converged (A3) and plays role of $V_j$ (A7)\n'
             'Large Δ = that query felt the perturbation', fontsize=10)
ax.legend()
ax.axhline(0, color='black', lw=0.8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# ── Figure 5: Large sentence — P_attn before/after + circle most changed ──────
fig5, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, P_plot, title in zip(
    axes,
    [P_attn_l, P_attn_lp],
    ['$P_{attn}$ Before flip  (large sentence)',
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

# Circle most changed cell on after heatmap
r_l, c_l = max_idx_l
axes[1].add_patch(mpatches.Circle(
    (c_l + 0.5, r_l + 0.5), 0.45,
    fill=False, edgecolor='orange', lw=3, zorder=5))
axes[1].text(c_l + 0.5, r_l - 0.3,
             f'most changed\n({sentence[r_l]}→{sentence[c_l]})',
             ha='center', va='bottom', fontsize=8,
             color='orange', fontweight='bold')

for g in range(4):
    mid = g * lattice_size + lattice_size / 2
    axes[0].text(N_large + 0.2, mid, lattice_labels[g],
                 ha='left', va='center', fontsize=7, color='red',
                 transform=axes[0].get_yaxis_transform())

plt.suptitle('Large Sentence (12 tokens, 4 lattice groups)\n'
             'Red lines = lattice boundaries  |  '
             'Orange circle = cell that gained/lost most attention after spin flip',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 6: O(N^2) cost ─────────────────────────────────────────────────────
fig6, axes = plt.subplots(1, 2, figsize=(13, 5))
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
                  'Each (i,j) pair converges independently (A3)\n'
                  'All-to-all emerges from $\\langle S_j \\rangle$ (A4)', fontsize=10)
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


print()
print("=" * 65)
print("MAPPING TABLE")
print("=" * 65)
rows = [
    ("Energy  E_ij = -mu*q*J_ij*<m_j>_i",      "Stability of query-key pair"),
    ("Inv. temp  beta = 1/sqrt(d)",              "Scaling factor"),
    ("Coordination number q=5",                  "Each spin sees 5 neighbours"),
    ("Converged <m_j>_i  (self-consistent)",     "Value Vj  (A7)"),
    ("Z_i = prod_j 2*cosh(beta*mu*q*J_ij)",     "Softmax denominator"),
    ("P_spin = exp(bmqJ) / 2*cosh(bmqJ)",       "Single-spin Boltzmann (microscopic)"),
    ("P_attn = exp(bmqJ) / Z_i  [emerges]",     "Attention weight (not applied)"),
    ("<M_i> = sum_j P_attn_ij * <m_j>_i",       "Attention output"),
    ("Spins / states",                           "Tokens"),
    ("J_ij = qi.kj / sqrt(d_k)",                "Similarity score"),
    ("beta->inf  (T->0, ordered)",               "Sharp / peaked attention"),
    ("beta->0   (T->inf, disordered)",           "Uniform / flat attention"),
    ("F_i = -(1/beta)*ln(Z_i)",                  "Log partition function"),
    ("All-to-all from <Sj>  (A4)",              "All-to-all attention (not designed)"),
]
print(f"  {'Mean-Field Ising':<45}  Attention")
print(f"  {'-'*45}  {'-'*30}")
for left, right in rows:
    print(f"  {left:<45}  {right}")
print()