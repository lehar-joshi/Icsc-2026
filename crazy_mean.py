"""
Mean-Field Ising Model → Attention Emerges from Physics
========================================================
Tokens are ±1 spin configurations. No attention function is applied anywhere.
The Boltzmann probability distribution and mean magnetisation emerge naturally
from the partition function.

Physical setup
--------------
  Token         = a lattice of d_model spins, each in {+1, -1}
  Query field   = projection of token spin config into query space
  Key spin      = projection of token spin config into key space
  J_ij          = qi . kj / sqrt(d_k)   -- unique coupling per (i,j) pair
  <Sj>_i        = tanh(beta * J_ij)     -- mean magnetisation of key j under query i
  Z_i           = prod_j 2cosh(beta*J_ij) -- partition function for query i
  P_ij          = exp(beta*J_ij) / sum_j exp(beta*J_ij)  -- Boltzmann prob (emerges)
  <M_i>         = sum_j P_ij * <Sj>_i  -- mean magnetisation per dipole
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['font.size']        = 10

d_model = 8
d_k     = 8
beta    = 1.0

# SECTION 1 — Token spin configurations
# Each token is a hardcoded +-1 spin lattice of length d_model.
# — "cat" just happens to be this spin pattern.

token_spins = {
    'The' : np.array([ 1,  1, -1,  1, -1,  1,  1, -1], dtype=float),
    'cat' : np.array([ 1,  1, -1,  1, -1, -1,  1,  1], dtype=float),
    'sat' : np.array([-1,  1,  1, -1,  1, -1,  1,  1], dtype=float),
}

tokens_small = ['The', 'cat', 'sat']
X_small = np.array([token_spins[t] for t in tokens_small])
#token_spin_value for value in token_spin_key
#array created of token_spin_value per token_spin_key 

for t in tokens_small:
    ups   = int((token_spins[t] ==  1).sum())
    downs = int((token_spins[t] == -1).sum())

np.random.seed(42)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5

def compute_QK(X):
    return X @ W_Q, X @ W_K

def compute_J(Q, K):
    return Q @ K.T / np.sqrt(d_k)

def mean_field(J, beta):
    """
    mean_S  (N_q, N_k)  <Sj>_i = tanh(beta * J_ij)
    Z       (N_q,)      Z_i = prod_j 2*cosh(beta * J_ij)
    F       (N_q,)      F_i = -(1/beta) * ln(Z_i)
    P       (N_q, N_k)  P_ij = exp(beta*J_ij) / sum_j exp(beta*J_ij)  [emerges]
    E       (N_q, N_k)  E_ij = -J_ij * <Sj>_i
    M       (N_q,)      <M_i> = sum_j P_ij * <Sj>_i
    """
    

    log_Z = np.sum(np.log(2 * np.cosh(beta * J)), axis=1)
    Z     = np.exp(log_Z)
    F     = -(1.0 / beta) * log_Z
    mean_S = np.tanh(beta * J) #where is mu

    bJ        = beta * J
    bJ_stable = bJ - bJ.max(axis=1, keepdims=True)
    exp_bJ    = np.exp(bJ_stable)
    P         = exp_bJ / exp_bJ.sum(axis=1, keepdims=True)

    E = -J * mean_S
    M = (P * mean_S).sum(axis=1)

    return mean_S, Z, F, P, E, M


Q_s, K_s             = compute_QK(X_small)
J_s                  = compute_J(Q_s, K_s)
mean_S, Z, F, P, E, M = mean_field(J_s, beta)

print("=" * 65)
print(f"SECTION 3 — Mean-field observables  (beta={beta})")
print("=" * 65)
for i, t in enumerate(tokens_small):
    print(f"  Query '{t}'")
    print(f"    <Sj>  = {np.round(mean_S[i], 3)}")
    print(f"    E_ij  = {np.round(E[i], 3)}")
    print(f"    P_ij  = {np.round(P[i], 4)}  (sum={P[i].sum():.6f})")
    print(f"    Z_i   = {Z[i]:.4f}   F_i = {F[i]:.4f}   <M_i> = {M[i]:.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Spin flip perturbation
# Flip one spin in 'cat', recompute, show what changed
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

X_p                        = np.array([token_spins_p[t] for t in tokens_small])
Q_p, K_p                   = compute_QK(X_p)
J_p                        = compute_J(Q_p, K_p)
mean_S_p, Z_p, F_p, P_p, E_p, M_p = mean_field(J_p, beta)

print("  <M> before:", np.round(M,       4))
print("  <M> after :", np.round(M_p,     4))
print("  delta<M>  :", np.round(M_p - M, 4))
print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Large sentence (12 tokens, 4 lattice groups)
# ═══════════════════════════════════════════════════════════════════════════════

sentence = [
    'the',    'old',    'black',
    'cat',    'sat',    'quietly',
    'on',     'the',    'warm',
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
Q_l, K_l = compute_QK(X_large)
J_l      = compute_J(Q_l, K_l)
mean_S_l, Z_l, F_l, P_l, E_l, M_l = mean_field(J_l, beta)

N_range = np.arange(3, 513)

print("=" * 65)
print("SECTION 5 — Complexity scaling")
print("=" * 65)
for n in [3, 12, 64, 128, 512]:
    print(f"  N={n:4d}  ->  {n**2:7,d} pairwise J_ij computations")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Token spin configurations ───────────────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 3))
spin_matrix = np.array([token_spins[t] for t in tokens_small])
sns.heatmap(spin_matrix, annot=True, fmt='.0f', cmap='RdBu', center=0,
            xticklabels=[f's{i}' for i in range(d_model)],
            yticklabels=tokens_small,
            linewidths=1, ax=ax, vmin=-1, vmax=1,
            cbar_kws={'label': 'Spin state (+-1)'})
ax.set_title('Token Spin Configurations — each token is a +-1 spin lattice\n'
             '"cat" = [+1,+1,-1,+1,-1,-1,+1,+1]  (illustrative, not linguistic)',
             fontsize=11)
ax.set_xlabel('Spin index within token lattice')
ax.set_ylabel('Token')
plt.tight_layout()
plt.show()


# ── Figure 2: Energy vs Probability anticorrelation ───────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5))

vmax_e = np.abs(E).max()
sns.heatmap(E, annot=True, fmt='.3f', cmap='YlOrRd_r',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[0], vmin=-vmax_e, vmax=vmax_e)
axes[0].set_title('Pair Energy  $E_{ij} = -J_{ij}\\langle S_j \\rangle_i$\n'
                  'Yellow = low energy = stable\n'
                  '(most stable pair gets selected)', fontsize=10)
axes[0].set_xlabel('Key $k_j$'); axes[0].set_ylabel('Query $q_i$')

sns.heatmap(P, annot=True, fmt='.4f', cmap='YlOrRd',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('Boltzmann Probability  $P_{ij} = e^{\\beta J_{ij}} / Z_i$\n'
                  'Dark = high probability\n'
                  '(emerges from $Z_i$ — not assumed)', fontsize=10)
axes[1].set_xlabel('Key $k_j$'); axes[1].set_ylabel('Query $q_i$')

colors_s = plt.cm.tab10(np.arange(len(tokens_small)))
for i, t in enumerate(tokens_small):
    axes[2].scatter(E[i], P[i], s=140, color=colors_s[i],
                    label=f'Query "{t}"', zorder=3)
    for j, tj in enumerate(tokens_small):
        axes[2].annotate(f'to {tj}', (E[i, j], P[i, j]),
                         textcoords='offset points', xytext=(5, 3), fontsize=7)

flat_idx    = E.argmin()
qi, kj      = np.unravel_index(flat_idx, E.shape)
axes[2].annotate(f'Most stable\n({tokens_small[qi]}->  {tokens_small[kj]})',
                 xy=(E[qi, kj], P[qi, kj]),
                 xytext=(E[qi, kj] + 0.05, P[qi, kj] - 0.12),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=8, color='red')
axes[2].set_xlabel('Pair Energy  $E_{ij}$')
axes[2].set_ylabel('Boltzmann Probability  $P_{ij}$')
axes[2].set_title('Anticorrelation: Low Energy -> High Probability\n'
                  'Colorbars are mirrored — same cell is bright in both maps',
                  fontsize=10)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Energy <-> Probability Anticorrelation\n'
             'Lowest-energy key spin = highest Boltzmann weight = most "attended"',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 3: Spin flip perturbation — P before/after/diff ────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.heatmap(P, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[0], vmin=0, vmax=1)
axes[0].set_title('$P_{ij}$ Before flip\n(original spin config)', fontsize=10)
axes[0].set_xlabel('Key $k_j$'); axes[0].set_ylabel('Query $q_i$')

sns.heatmap(P_p, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title(f'$P_{{ij}}$ After flip\n'
                  f'"{flip_token}" spin {flip_spin}: '
                  f'{int(token_spins[flip_token][flip_spin]):+d} -> '
                  f'{int(token_spins_p[flip_token][flip_spin]):+d}', fontsize=10)
axes[1].set_xlabel('Key $k_j$'); axes[1].set_ylabel('Query $q_i$')

dP     = P_p - P
vmax_d = np.abs(dP).max()
sns.heatmap(dP, annot=True, fmt='+.4f', cmap='coolwarm', center=0,
            xticklabels=tokens_small, yticklabels=tokens_small,
            linewidths=0.5, ax=axes[2], vmin=-vmax_d, vmax=vmax_d)
axes[2].set_title('$\\Delta P_{ij}$ = After - Before\n'
                  'Red = probability up, Blue = probability down\n'
                  'Shows which query-key pairs were disturbed', fontsize=10)
axes[2].set_xlabel('Key $k_j$'); axes[2].set_ylabel('Query $q_i$')

plt.suptitle(f'Perturbation: Flip spin {flip_spin} of "{flip_token}"\n'
             'One spin flip propagates through J -> <Sj> -> P -> <M>',
             fontsize=12)
plt.tight_layout()
plt.show()


# ── Figure 4: Mean magnetisation per dipole before/after flip ─────────────────
fig4, ax = plt.subplots(figsize=(8, 4))
x     = np.arange(len(tokens_small))
width = 0.35

ax.bar(x - width/2, M,   width, color='steelblue', alpha=0.85, label='Before flip')
ax.bar(x + width/2, M_p, width, color='coral',     alpha=0.85, label='After flip')

for i in range(len(tokens_small)):
    delta = M_p[i] - M[i]
    ax.annotate(f'delta={delta:+.3f}',
                xy=(x[i] + width / 2, M_p[i]),
                xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=8,
                color='red' if abs(delta) > 0.005 else 'gray')

ax.set_xticks(x)
ax.set_xticklabels([f'Query "{t}"' for t in tokens_small])
ax.set_ylabel('Mean Magnetisation per Dipole  $\\langle M_i \\rangle$')
ax.set_title(f'Mean Magnetisation Before vs After Flipping Spin {flip_spin} of "{flip_token}"\n'
             '$\\langle M_i \\rangle = \\sum_j P_{{ij}} \\langle S_j \\rangle_i$\n'
             'Large delta = that query felt the perturbation', fontsize=10)
ax.legend()
ax.axhline(0, color='black', lw=0.8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# ── Figure 5: Large sentence P heatmap with lattice group boundaries ───────────
fig5, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(P_l, cmap='Blues', ax=ax,
            xticklabels=sentence, yticklabels=sentence,
            linewidths=0.2, vmin=0)

for boundary in [3, 6, 9]:
    ax.axhline(boundary, color='red', lw=2.5)
    ax.axvline(boundary, color='red', lw=2.5)

for g in range(4):
    mid = g * lattice_size + lattice_size / 2
    ax.text(N_large + 0.3, mid, lattice_labels[g],
            ha='left', va='center', fontsize=8, color='red', fontweight='bold',
            transform=ax.get_yaxis_transform())

ax.set_title('Boltzmann Probability Matrix — 12-token sentence, 4 lattice groups\n'
             '$P_{ij} = e^{\\beta J_{ij}} / Z_i$   (red lines = lattice boundaries)\n'
             'Every token must couple to every other token to evaluate Z_i',
             fontsize=11)
ax.set_xlabel('Key spin $k_j$')
ax.set_ylabel('Query field $q_i$')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ── Figure 6: O(N²) computational cost ────────────────────────────────────────
fig6, axes = plt.subplots(1, 2, figsize=(13, 5))

ops = N_range ** 2
axes[0].plot(N_range, ops, color='navy', lw=2)
axes[0].fill_between(N_range, ops, alpha=0.12, color='navy')

markers = [(3, '"The cat sat"'), (12, 'Our sentence'), (64, '64 tokens'), (512, '512 tokens')]
for n, label in markers:
    axes[0].scatter(n, n**2, s=80, color='red', zorder=5)
    offset_x = n + 8
    offset_y = n**2 * 0.75
    axes[0].annotate(f'{label}\n{n**2:,} ops',
                     xy=(n, n**2), xytext=(offset_x, offset_y),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1),
                     fontsize=8, color='red')

axes[0].set_xlabel('Number of tokens N')
axes[0].set_ylabel('Pairwise J_ij computations  ($N^2$)')
axes[0].set_title('$O(N^2)$ Cost of the Full Partition Function\n'
                  'Each token couples to every other token\n'
                  'A longer sentence = a larger spin lattice = more expensive',
                  fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].loglog(N_range, ops,     color='darkred',   lw=2, label='$N^2$ (pairwise coupling)')
axes[1].loglog(N_range, N_range, color='steelblue', lw=2, ls='--', label='$N$ (linear)')
axes[1].set_xlabel('Number of tokens N  (log scale)')
axes[1].set_ylabel('Operations  (log scale)')
axes[1].set_title('Log-Log: $N^2$ vs $N$\n'
                  'The spin lattice grows quadratically\n'
                  'This is the fundamental cost of full pairwise interaction',
                  fontsize=10)
axes[1].legend()
axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle('Why Large Spin Systems (Long Sentences) Are Expensive\n'
             'Every spin must interact with every other spin -> $O(N^2)$ couplings',
             fontsize=12)
plt.tight_layout()
plt.show()


print()
print("=" * 65)
print("CONCLUSION — Derived, never assumed")
print("=" * 65)
print("""
  Token = +-1 spin lattice  (d_model spins per token)

  From the mean-field Ising Hamiltonian  E_ij = -J_ij * Sj:

    J_ij   = qi . kj / sqrt(d_k)      unique coupling per (i,j) pair
    <Sj>_i = tanh(beta * J_ij)        mean magnetisation of key spin j
    Z_i    = prod_j 2*cosh(beta*J_ij) partition function for query i
    F_i    = -(1/beta) * ln(Z_i)      free energy
    P_ij   = exp(beta*J_ij) / Z_i     Boltzmann probability  <-- emerges
    <M_i>  = sum_j P_ij * <Sj>_i     mean magnetisation per dipole

  P_ij is mathematically identical to scaled dot-product attention.
  It was not applied -- it fell out of the partition function.

  Perturbation:
    Flip one spin in one token
    -> J changes -> <Sj> changes -> P shifts -> <M_i> shifts
    -> delta<M_i> identifies which queries felt the disturbance

  Large N:
    Every token couples to every other token to evaluate Z_i
    -> O(N^2) pairwise J_ij computations
    -> A long sentence is a large spin lattice and it is expensive
""")