"""
Mean-Field Ising Model → Attention Mechanism (Corrected)
=========================================================

CORE CLAIM
----------
  exp(β·μ·q·J_ij) / Σ_j exp(β·μ·q·J_ij)  ·  m_j
= exp(q_i·k_j/√d_k)  / Σ_j exp(q_i·k_j/√d_k)  ·  V_j

When β·μ·q = 1, the Boltzmann weights ARE softmax attention weights.
The self-consistent equilibrium magnetisation m_j serves as Value.

ASSUMPTIONS
-----------
A1. μ = 1          Magnetic moment.
A2. B_ext = 0      For attention equivalence. B≠0 explored for phase transition.
A3. J_ij = q_i·k_j/√d_k   Coupling = dot product similarity.
A4. q = 2 (3 tokens), q = 4 (8 tokens)   Coordination number ~ neighbours
    within correlation length. β set so β·μ·q = 1 for exact mapping.
A5. W_Q ≠ W_K      Asymmetric projections → J_ij ≠ J_ji in general.
A6. Self-consistent global mean-field for Values:
        m_i = tanh(β·μ·(q · Σ_j J_ij · m_j + B_ext))
    Gives query-INDEPENDENT m vector. m_j is the value of token j.
A7. Two probability objects:
    P_spin_ij = σ(2·β·μ·q·J_ij)           Single-spin Boltzmann (microscopic)
    P_attn_ij = softmax_j(β·μ·q·J_ij)     Emergent attention weight
A8. Order parameter: M = (1/N)·Σ_i |m_i|  (magnetisation per dipole)
    B=0  → phase transition at β_c (para ↔ ferro = uniform ↔ peaked attention)
    B≠0  → always magnetised, smooth crossover (no true transition)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

np.random.seed(42)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'font.size':        10,
    'figure.dpi':       150,
    'savefig.bbox':     'tight',
    'savefig.dpi':      150,
})

# =============================================================================
# SETUP
# =============================================================================
mu       = 1.0
d_model  = 8
d_k      = 8
tol      = 1e-8
max_iter = 20000

# Projection matrices (A5)
np.random.seed(42)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_k) * 0.5

# ── 3-token system ────────────────────────────────────────────────────────────
tokens_3 = ['The', 'cat', 'sat']
token_spins_3 = {
    'The': np.array([ 1,  1, -1,  1, -1,  1,  1, -1], dtype=float),
    'cat': np.array([ 1,  1, -1,  1, -1, -1,  1,  1], dtype=float),
    'sat': np.array([-1,  1,  1, -1,  1, -1,  1,  1], dtype=float),
}
X_3 = np.array([token_spins_3[t] for t in tokens_3])
q_3    = 2                        # A4: coordination number for 3 tokens
beta_3 = 1.0 / (mu * q_3)        # β·μ·q = 1 → exact softmax match

# ── 8-token system (2×4 grid) ─────────────────────────────────────────────────
sentence_8 = ['the', 'old', 'cat', 'sat', 'on', 'warm', 'mat', 'today']
np.random.seed(7)
X_8 = np.array([np.random.choice([-1.0, 1.0], size=d_model) for _ in sentence_8])
q_8    = 4                        # A4: coordination number for 8 tokens
beta_8 = 1.0 / (mu * q_8)        # β·μ·q = 1

CAT_IDX_8  = sentence_8.index('cat')
FLIP_SPIN  = 2

print("=" * 70)
print("SETUP")
print("=" * 70)
print(f"  mu = {mu},  d_model = {d_model},  d_k = {d_k}")
print(f"  3-token: q = {q_3},  beta = {beta_3:.4f}  (beta*mu*q = {beta_3*mu*q_3})")
print(f"  8-token: q = {q_8},  beta = {beta_8:.4f}  (beta*mu*q = {beta_8*mu*q_8})")
print()


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_J(X):
    """J_ij = q_i . k_j / sqrt(d_k)  (coupling matrix from QK similarity)"""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    return Q @ K.T / np.sqrt(d_k), V


def standard_softmax(logits):
    """Standard attention softmax over axis=1 (keys)."""
    s = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def boltzmann_attention(J, beta, mu, q):
    """
    Boltzmann attention weights from mean-field Ising:
        P_ij = exp(beta*mu*q*J_ij) / sum_j exp(beta*mu*q*J_ij)

    When beta*mu*q = 1, this is exactly softmax(J).
    """
    logits = beta * mu * q * J
    return standard_softmax(logits)


def P_spin_single(J, beta, mu, q):
    """
    Single-spin Boltzmann probability P(S_j = +1 | query i):
        P = sigma(2*beta*mu*q*J_ij)
    This is microscopic -- tracks spin state, not attention allocation.
    """
    return 1.0 / (1.0 + np.exp(-2.0 * beta * mu * q * J))


def solve_global_mf(J, beta, mu, q, B_ext=0.0, tol=1e-8, max_iter=20000):
    """
    Coupled mean-field equation (GLOBAL, query-independent):
        m_i = tanh(beta * mu * (q * sum_j J_ij * m_j  +  B_ext))

    Returns
    -------
    m      : (N,) equilibrium magnetisation per token = Value vector
    iters  : iterations to convergence
    """
    N = J.shape[0]
    # Seed with small positive values (broken symmetry for B=0)
    m = np.full(N, 0.1)
    for it in range(max_iter):
        h = q * (J @ m) + B_ext
        m_new = np.tanh(beta * mu * h)
        if np.abs(m_new - m).max() < tol:
            return m_new, it + 1
        m = m_new
    return m, max_iter


def order_parameter(m):
    """Magnetisation per dipole: M = (1/N) sum |m_i|"""
    return np.mean(np.abs(m))


# =============================================================================
# SECTION 1 -- BOLTZMANN = SOFTMAX EQUIVALENCE (3 tokens)
# =============================================================================
print("=" * 70)
print("SECTION 1 -- Boltzmann = Softmax Equivalence")
print("=" * 70)

J_3,V = compute_J(X_3)
P_softmax_3 = standard_softmax(J_3)
attention_1= P_softmax_3@V                        # standard attention
P_boltz_3   = boltzmann_attention(J_3, beta_3, mu, q_3)     # Ising Boltzmann

residual = np.abs(P_boltz_3 - P_softmax_3).max()
print(attention_1)
print(f"  J matrix (3x3):\n{np.round(J_3, 4)}")
print(f"  beta*mu*q = {beta_3 * mu * q_3:.1f}")
print(f"  Max |P_Boltzmann - P_softmax| = {residual:.2e}")
print(f"  -> {'EXACT MATCH' if residual < 1e-14 else 'MISMATCH'}")
print()

# ── Figure 1: Equivalence proof ───────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 4.5))

kw = dict(annot=True, fmt='.4f', linewidths=0.5,
          xticklabels=tokens_3, yticklabels=tokens_3, vmin=0, vmax=1)

sns.heatmap(attention_1, cmap='Blues', ax=axes[0], **kw)
axes[0].set_title('Standard Softmax Attention\n'
                  r'$P_{ij} = \frac{e^{J_{ij}}}{\sum_j e^{J_{ij}}}$'
                  '\nwhere $J_{ij} = q_i \\cdot k_j / \\sqrt{d_k}$',
                  fontsize=9)
axes[0].set_xlabel('Key'); axes[0].set_ylabel('Query')

sns.heatmap(P_boltz_3, cmap='Greens', ax=axes[1], **kw)
axes[1].set_title('Boltzmann Attention (Mean-Field Ising)\n'
                  r'$P_{ij} = \frac{e^{\beta \mu q J_{ij}}}{\sum_j e^{\beta \mu q J_{ij}}}$'
                  f'\nbeta*mu*q = {beta_3*mu*q_3:.0f} -> identical',
                  fontsize=9)
axes[1].set_xlabel('Key'); axes[1].set_ylabel('Query')

diff = P_boltz_3 - P_softmax_3
vd = max(np.abs(diff).max(), 1e-16)
sns.heatmap(diff, cmap='coolwarm', center=0, ax=axes[2],
            annot=True, fmt='.1e', linewidths=0.5,
            xticklabels=tokens_3, yticklabels=tokens_3,
            vmin=-vd, vmax=vd)
axes[2].set_title('Difference (Boltzmann - Softmax)\n'
                  'Machine-zero everywhere\n'
                  '-> Boltzmann weights ARE attention weights', fontsize=9)
axes[2].set_xlabel('Key'); axes[2].set_ylabel('Query')

fig1.suptitle('Core Result: Mean-Field Ising Boltzmann Weights = Softmax Attention\n'
              f'3 tokens, q={q_3}, beta=1/(mu*q)={beta_3:.2f}, d_k={d_k}',
              fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
print("  [Figure 1 saved: Boltzmann = Softmax equivalence]")


# =============================================================================
# SECTION 2 -- SELF-CONSISTENT m AS VALUES  +  ENERGY LANDSCAPE
# =============================================================================
print()
print("=" * 70)
print("SECTION 2 -- Self-Consistent Magnetisation = Values")
print("=" * 70)

# B=0: values from pure spin-spin interaction
m_3_B0, it_B0 = solve_global_mf(J_3, beta_3, mu, q_3, B_ext=0.0)
# B!=0: external field breaks symmetry -> always magnetised
m_3_B1, it_B1 = solve_global_mf(J_3, beta_3, mu, q_3, B_ext=0.5)

# Attention output: o_i = sum_j P_attn_ij * m_j
o_3_B0 = P_boltz_3 @ m_3_B0
o_3_B1 = P_boltz_3 @ m_3_B1

# Pair energy: E_ij = -mu * q * J_ij * m_j  (energy of query i attending to key j)
E_3 = -mu * q_3 * J_3 * m_3_B0[np.newaxis, :]  # broadcast m_j across queries

print(f"  B=0:  m = {np.round(m_3_B0, 6)}  ({it_B0} iters)")
print(f"        M (order param) = {order_parameter(m_3_B0):.6f}")
print(f"        output = {np.round(o_3_B0, 6)}")
print(f"  B=0.5: m = {np.round(m_3_B1, 6)}  ({it_B1} iters)")
print(f"        M (order param) = {order_parameter(m_3_B1):.6f}")
print(f"        output = {np.round(o_3_B1, 6)}")
print()

# ── Figure 2: Values, Energy, P_spin, Attention output ───────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(17, 9))

# Row 1: physical observables
# (a) P_attn
sns.heatmap(P_boltz_3, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_3, yticklabels=tokens_3,
            linewidths=0.5, ax=axes[0, 0], vmin=0, vmax=1)
axes[0, 0].set_title('$P_{attn}$ (Boltzmann = Softmax)\n'
                      'Which key does query attend to?', fontsize=9)
axes[0, 0].set_xlabel('Key'); axes[0, 0].set_ylabel('Query')

# (b) Pair energy
vmax_e = np.abs(E_3).max()
if vmax_e < 1e-10:
    vmax_e = 1.0
sns.heatmap(E_3, annot=True, fmt='.3f', cmap='YlOrRd_r',
            xticklabels=tokens_3, yticklabels=tokens_3,
            linewidths=0.5, ax=axes[0, 1], vmin=-vmax_e, vmax=vmax_e)
axes[0, 1].set_title('Pair Energy $E_{ij} = -\\mu q J_{ij} m_j$\n'
                      'Low energy = stable = high attention', fontsize=9)
axes[0, 1].set_xlabel('Key'); axes[0, 1].set_ylabel('Query')

# (c) P_spin (microscopic)
P_spin_3 = P_spin_single(J_3, beta_3, mu, q_3)
sns.heatmap(P_spin_3, annot=True, fmt='.4f', cmap='Greens',
            xticklabels=tokens_3, yticklabels=tokens_3,
            linewidths=0.5, ax=axes[0, 2], vmin=0, vmax=1)
axes[0, 2].set_title('$P_{spin}(S_j=+1|i) = \\sigma(2\\beta\\mu q J_{ij})$\n'
                      'Microscopic spin-state probability\n'
                      '(does NOT sum to 1 over j)', fontsize=9)
axes[0, 2].set_xlabel('Key'); axes[0, 2].set_ylabel('Query')

# Row 2: values and outputs
# (d) m values at B=0 vs B!=0
x = np.arange(len(tokens_3))
w = 0.35
axes[1, 0].bar(x - w/2, m_3_B0, w, color='steelblue', alpha=0.85, label='B=0')
axes[1, 0].bar(x + w/2, m_3_B1, w, color='coral',     alpha=0.85, label='B=0.5')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(tokens_3)
axes[1, 0].set_ylabel('$m_j$ (equilibrium magnetisation)')
axes[1, 0].set_title('Values $m_j$ from self-consistent MF\n'
                      '$m_i = \\tanh(\\beta\\mu(q \\sum_j J_{ij} m_j + B))$\n'
                      'Query-independent -> serves as $V_j$', fontsize=9)
axes[1, 0].legend()
axes[1, 0].axhline(0, color='black', lw=0.5)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# (e) Attention output
axes[1, 1].bar(x - w/2, o_3_B0, w, color='steelblue', alpha=0.85, label='B=0')
axes[1, 1].bar(x + w/2, o_3_B1, w, color='coral',     alpha=0.85, label='B=0.5')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(tokens_3)
axes[1, 1].set_ylabel('$o_i = \\sum_j P_{attn,ij} \\cdot m_j$')
axes[1, 1].set_title('Attention Output\n'
                      '$o_i = \\sum_j P_{attn,ij} \\cdot m_j$\n'
                      'Weighted sum of equilibrium values', fontsize=9)
axes[1, 1].legend()
axes[1, 1].axhline(0, color='black', lw=0.5)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# (f) Energy minimisation = attention maximisation scatter
colors = ['#e74c3c', '#2196F3', '#4CAF50']
for i, (t, c) in enumerate(zip(tokens_3, colors)):
    axes[1, 2].scatter(E_3[i], P_boltz_3[i], c=c, s=100, zorder=5,
                        label=f'Query "{t}"')
    for j, tk in enumerate(tokens_3):
        axes[1, 2].annotate(f'->{tk}', (E_3[i, j], P_boltz_3[i, j]),
                            textcoords='offset points', xytext=(5, 5), fontsize=7)
axes[1, 2].set_xlabel('Pair energy $E_{ij}$')
axes[1, 2].set_ylabel('Attention weight $P_{attn,ij}$')
axes[1, 2].set_title('Energy Minimisation = Max Attention\n'
                      'Lower energy -> higher Boltzmann weight\n'
                      '= magnetic alignment = domain formation', fontsize=9)
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

fig2.suptitle('Mean-Field Observables: Energy, Spin States, Values, and Output\n'
              f'3 tokens, q={q_3}, beta={beta_3:.2f}, mu={mu}',
              fontsize=12, fontweight='bold')
plt.tight_layout()
print("  [Figure 2 saved: Observables]")
plt.show()


# =============================================================================
# SECTION 3 -- PHASE TRANSITION  (beta sweep, B=0 vs B!=0)
# =============================================================================
print()
print("=" * 70)
print("SECTION 3 -- Phase Transition: Paramagnetism <-> Ferromagnetism")
print("=" * 70)

# Critical beta: beta_c * q * lambda_max(J) = 1  (mean-field criterion)
J_sym = 0.5 * (J_3 + J_3.T)
eigvals = np.linalg.eigvalsh(J_sym)
lambda_max = eigvals.max()
beta_c = 1.0 / (q_3 * lambda_max) if lambda_max > 0 else np.inf

print(f"  J eigenvalues (symmetrised): {np.round(eigvals, 4)}")
print(f"  lambda_max = {lambda_max:.4f}")
print(f"  beta_c = 1/(q*lambda_max) = {beta_c:.4f}")
print(f"  -> below beta_c: paramagnetic (uniform attention)")
print(f"  -> above beta_c: ferromagnetic (peaked attention)")
print()

betas = np.linspace(0.005, 1.5, 300)
M_B0      = []
M_B05     = []
entropy_B0 = []

for b in betas:
    m_b0, _ = solve_global_mf(J_3, b, mu, q_3, B_ext=0.0)
    m_b1, _ = solve_global_mf(J_3, b, mu, q_3, B_ext=0.5)
    M_B0.append(order_parameter(m_b0))
    M_B05.append(order_parameter(m_b1))
    P = boltzmann_attention(J_3, b, mu, q_3)
    H = -np.sum(P * np.log(P + 1e-30)) / len(tokens_3)
    entropy_B0.append(H)

M_B0       = np.array(M_B0)
M_B05      = np.array(M_B05)
entropy_B0 = np.array(entropy_B0)

# ── Figure 3: Phase transition ────────────────────────────────────────────────
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(betas, M_B0,  color='navy', lw=2.5, label='$B_{ext}=0$ (pure spin-spin)')
ax1.plot(betas, M_B05, color='coral', lw=2.5, label='$B_{ext}=0.5$ (external field)')
ax1.axvline(beta_c, color='gray', ls='--', lw=1.5,
            label=f'$\\beta_c = {beta_c:.3f}$')
ax1.axvline(beta_3, color='green', ls=':', lw=2,
            label=f'$\\beta = 1/(\\mu q) = {beta_3:.2f}$ (attention equiv.)')
ax1.set_ylabel('Order parameter\n$M = \\frac{1}{N}\\sum_i |m_i|$\n'
               '(magnetisation per dipole)')
ax1.set_title('Phase Transition: Paramagnetism <-> Ferromagnetism\n'
              'B=0: spontaneous symmetry breaking at $\\beta_c$  |  '
              'B!=0: always magnetised (no true transition)\n'
              'Maps to: uniform attention (para) -> peaked attention (ferro)',
              fontsize=10, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.02, 1.02)

ax1.annotate('PARAMAGNETIC\n(uniform attention\nflat softmax)',
             xy=(beta_c * 0.3, 0.05), fontsize=9, color='navy',
             ha='center', style='italic')
ax1.annotate('FERROMAGNETIC\n(peaked attention\nsharp softmax)',
             xy=(min(beta_c * 2.5, 1.4), 0.85), fontsize=9, color='navy',
             ha='center', style='italic')

ax2.plot(betas, entropy_B0, color='purple', lw=2.5)
ax2.axvline(beta_c, color='gray', ls='--', lw=1.5)
ax2.axvline(beta_3, color='green', ls=':', lw=2)
max_entropy = np.log(len(tokens_3))
ax2.axhline(max_entropy, color='orange', ls='-.', lw=1,
            label=f'Max entropy = ln({len(tokens_3)}) = {max_entropy:.2f} (uniform)')
ax2.set_xlabel('Inverse temperature $\\beta$ (= $1/T$)')
ax2.set_ylabel('Avg attention entropy\n$H = -\\sum_j P_{ij} \\ln P_{ij}$')
ax2.set_title('Attention Entropy vs Temperature\n'
              'High entropy = uniform attention = paramagnetic  |  '
              'Low entropy = peaked = ferromagnetic', fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("  [Figure 3 saved: Phase transition]")


# ── Figure 4: P_attn heatmaps at 3 temperatures ──────────────────────────────
beta_low  = beta_c * 0.3
beta_mid  = beta_c * 1.0
beta_high = beta_c * 4.0

fig4, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, b, label in zip(axes,
                         [beta_low, beta_mid, beta_high],
                         ['Paramagnetic\n(high T, low beta)',
                          f'Critical\n(beta ~ beta_c = {beta_c:.3f})',
                          'Ferromagnetic\n(low T, high beta)']):
    P = boltzmann_attention(J_3, b, mu, q_3)
    sns.heatmap(P, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=tokens_3, yticklabels=tokens_3,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title(f'{label}\nbeta = {b:.4f}', fontsize=9)
    ax.set_xlabel('Key'); ax.set_ylabel('Query')

fig4.suptitle('Attention Weights at Three Temperatures\n'
              'Uniform (para) -> Peaked (ferro) as beta increases',
              fontsize=12, fontweight='bold')
plt.tight_layout()
print("  [Figure 4 saved: Attention at 3 temperatures]")
plt.show()

# =============================================================================
# SECTION 4 -- SPIN FLIP PERTURBATION (3 tokens)
# =============================================================================
print()
print("=" * 70)
print("SECTION 4 -- Spin Flip Perturbation (3 tokens)")
print("=" * 70)

flip_token = 'cat'
flip_idx   = tokens_3.index(flip_token)
flip_dim   = 2

X_3_flip = X_3.copy()
old_spin = X_3_flip[flip_idx, flip_dim]
X_3_flip[flip_idx, flip_dim] *= -1
new_spin = X_3_flip[flip_idx, flip_dim]

J_3_flip      = compute_J(X_3_flip)
P_3_after     = boltzmann_attention(J_3_flip, beta_3, mu, q_3)
m_3_after, _  = solve_global_mf(J_3_flip, beta_3, mu, q_3, B_ext=0.0)
o_3_after     = P_3_after @ m_3_after

dP_3     = P_3_after - P_boltz_3
dJ_3     = J_3_flip - J_3
max_idx3 = np.unravel_index(np.abs(dP_3).argmax(), dP_3.shape)

print(f"  Flip: '{flip_token}' spin {flip_dim}: {int(old_spin):+d} -> {int(new_spin):+d}")
print(f"  Max |dJ| = {np.abs(dJ_3).max():.6f}")
print(f"  Max |dP_attn| = {np.abs(dP_3).max():.6f}  "
      f"at ({tokens_3[max_idx3[0]]} -> {tokens_3[max_idx3[1]]})")
print(f"  Output before: {np.round(o_3_B0, 6)}")
print(f"  Output after:  {np.round(o_3_after, 6)}")
print()

# ── Figure 5: Spin flip (3 tokens) ───────────────────────────────────────────
fig5, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.heatmap(P_boltz_3, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_3, yticklabels=tokens_3,
            linewidths=0.5, ax=axes[0], vmin=0, vmax=1)
axes[0].set_title('$P_{attn}$ Before Flip', fontsize=10)
axes[0].set_xlabel('Key'); axes[0].set_ylabel('Query')

sns.heatmap(P_3_after, annot=True, fmt='.4f', cmap='Blues',
            xticklabels=tokens_3, yticklabels=tokens_3,
            linewidths=0.5, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title(f'$P_{{attn}}$ After Flip\n'
                  f'"{flip_token}" spin {flip_dim}: '
                  f'{int(old_spin):+d} -> {int(new_spin):+d}', fontsize=10)
axes[1].set_xlabel('Key'); axes[1].set_ylabel('Query')

# Circle most changed cell
r3, c3 = max_idx3
axes[1].add_patch(mpatches.Circle(
    (c3 + 0.5, r3 + 0.5), 0.45,
    fill=False, edgecolor='red', lw=3, zorder=5))

vmax_d3 = max(np.abs(dP_3).max(), 1e-10)
sns.heatmap(dP_3, annot=True, fmt='+.4f', cmap='coolwarm', center=0,
            xticklabels=tokens_3, yticklabels=tokens_3,
            linewidths=0.5, ax=axes[2], vmin=-vmax_d3, vmax=vmax_d3)
axes[2].set_title('$\\Delta P_{attn}$ = After - Before\n'
                  'Red = attention increased | Blue = decreased', fontsize=10)
axes[2].set_xlabel('Key'); axes[2].set_ylabel('Query')
axes[2].add_patch(mpatches.Circle(
    (c3 + 0.5, r3 + 0.5), 0.45,
    fill=False, edgecolor='red', lw=3, zorder=5))

fig5.suptitle(f'Spin Flip Perturbation: "{flip_token}" spin {flip_dim} -> '
              f'Most changed: ({tokens_3[max_idx3[0]]} -> {tokens_3[max_idx3[1]]})  '
              f'dP = {dP_3[max_idx3]:+.6f}',
              fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
print("  [Figure 5 saved: 3-token spin flip]")


# =============================================================================
# SECTION 5 -- 8-TOKEN SYSTEM: SPIN FLIP ON 2x4 GRID
# =============================================================================
print()
print("=" * 70)
print("SECTION 5 -- 8-Token Spin Flip (2x4 lattice)")
print("=" * 70)

J_8        = compute_J(X_8)
P_8_before = boltzmann_attention(J_8, beta_8, mu, q_8)
m_8_before, _ = solve_global_mf(J_8, beta_8, mu, q_8, B_ext=0.0)

# Also verify equivalence for 8-token system
P_8_softmax = standard_softmax(J_8)
res_8 = np.abs(P_8_before - P_8_softmax).max()
print(f"  8-token equivalence check: Max |P_boltz - P_softmax| = {res_8:.2e}")

# Flip
X_8_flip = X_8.copy()
old_spin_8 = X_8_flip[CAT_IDX_8, FLIP_SPIN]
X_8_flip[CAT_IDX_8, FLIP_SPIN] *= -1
new_spin_8 = X_8_flip[CAT_IDX_8, FLIP_SPIN]

J_8_flip       = compute_J(X_8_flip)
P_8_after      = boltzmann_attention(J_8_flip, beta_8, mu, q_8)
m_8_after, _   = solve_global_mf(J_8_flip, beta_8, mu, q_8, B_ext=0.0)

dP_8     = P_8_after - P_8_before
max_idx8 = np.unravel_index(np.abs(dP_8).argmax(), dP_8.shape)

print(f"  Flip: '{sentence_8[CAT_IDX_8]}' spin {FLIP_SPIN}: "
      f"{int(old_spin_8):+d} -> {int(new_spin_8):+d}")
print(f"  Max |dP_attn| = {np.abs(dP_8).max():.6f}  "
      f"at ({sentence_8[max_idx8[0]]} -> {sentence_8[max_idx8[1]]})")
print(f"  M before = {order_parameter(m_8_before):.6f}")
print(f"  M after  = {order_parameter(m_8_after):.6f}")
print()

# Lattice group boundaries (2x4 grid: groups of 2)
lattice_boundaries = [2, 4, 6]
lattice_labels = ['Lattice 1\n(the,old)', 'Lattice 2\n(cat,sat)',
                  'Lattice 3\n(on,warm)', 'Lattice 4\n(mat,today)']

# ── Figure 6: 8-token P_attn before/after ────────────────────────────────────
fig6, axes = plt.subplots(1, 3, figsize=(24, 6.5))

for ax, P_plot, title in zip(
    axes[:2],
    [P_8_before, P_8_after],
    ['$P_{attn}$ Before Flip',
     f'$P_{{attn}}$ After Flip - "{sentence_8[CAT_IDX_8]}" spin {FLIP_SPIN} flipped']
):
    sns.heatmap(P_plot, cmap='Blues', ax=ax,
                xticklabels=sentence_8, yticklabels=sentence_8,
                linewidths=0.2, vmin=0, vmax=P_8_before.max())
    for b in lattice_boundaries:
        ax.axhline(b, color='red', lw=2)
        ax.axvline(b, color='red', lw=2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Key $k_j$'); ax.set_ylabel('Query $q_i$')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

# Circle most changed on "after"
r8, c8 = max_idx8
axes[1].add_patch(mpatches.Circle(
    (c8 + 0.5, r8 + 0.5), 0.45,
    fill=False, edgecolor='orange', lw=3, zorder=5))
axes[1].text(c8 + 0.5, r8 - 0.15,
             f'most changed\n({sentence_8[r8]}->{sentence_8[c8]})',
             ha='center', va='bottom', fontsize=8,
             color='orange', fontweight='bold')

# Lattice labels on the "before" plot
for g in range(4):
    mid = g * 2 + 1
    axes[0].text(len(sentence_8) + 0.3, mid, lattice_labels[g],
                 ha='left', va='center', fontsize=7, color='red',
                 transform=axes[0].get_yaxis_transform())

# Difference heatmap
vmax_d8 = max(np.abs(dP_8).max(), 1e-10)
sns.heatmap(dP_8, cmap='coolwarm', center=0, ax=axes[2],
            xticklabels=sentence_8, yticklabels=sentence_8,
            linewidths=0.2, vmin=-vmax_d8, vmax=vmax_d8,
            annot=True, fmt='+.3f', annot_kws={'size': 7})
for b in lattice_boundaries:
    axes[2].axhline(b, color='red', lw=2)
    axes[2].axvline(b, color='red', lw=2)
axes[2].set_title('$\\Delta P_{attn}$ = After - Before\n'
                  'How does a single spin flip propagate\n'
                  'through the lattice?', fontsize=10)
axes[2].set_xlabel('Key $k_j$'); axes[2].set_ylabel('Query $q_i$')
plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
plt.setp(axes[2].get_yticklabels(), rotation=0)
axes[2].add_patch(mpatches.Circle(
    (c8 + 0.5, r8 + 0.5), 0.45,
    fill=False, edgecolor='red', lw=3, zorder=5))

fig6.suptitle(f'8-Token Lattice (2x4): Spin Flip Propagation\n'
              f'Red lines = lattice boundaries  |  '
              f'Flip "{sentence_8[CAT_IDX_8]}" spin {FLIP_SPIN}  |  '
              f'Most changed: ({sentence_8[r8]} -> {sentence_8[c8]})  '
              f'dP = {dP_8[max_idx8]:+.6f}',
              fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
print("  [Figure 6 saved: 8-token spin flip heatmaps]")


# ── Figure 7: Magnetisation per dipole -- before/after flip ──────────────────
fig7, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-token magnetisation
x8 = np.arange(len(sentence_8))
w8 = 0.35
axes[0].bar(x8 - w8/2, np.abs(m_8_before), w8, color='steelblue', alpha=0.85,
            label='Before flip')
axes[0].bar(x8 + w8/2, np.abs(m_8_after),  w8, color='coral', alpha=0.85,
            label='After flip')
axes[0].set_xticks(x8)
axes[0].set_xticklabels(sentence_8, rotation=45, ha='right')
axes[0].set_ylabel('$|m_j|$ (magnetisation per dipole)')
axes[0].set_title('Per-Token Magnetisation (Order Parameter)\n'
                  '$m_j$ = equilibrium value of token $j$\n'
                  'Also serves as Value $V_j$ in attention', fontsize=9)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].annotate('<- flipped',
                 xy=(CAT_IDX_8 + w8/2, np.abs(m_8_after[CAT_IDX_8])),
                 xytext=(CAT_IDX_8 + 1.5,
                         np.abs(m_8_after[CAT_IDX_8]) + 0.05),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 color='red', fontsize=9)

# Attention output per query
o_8_before = P_8_before @ m_8_before
o_8_after  = P_8_after  @ m_8_after
axes[1].bar(x8 - w8/2, o_8_before, w8, color='steelblue', alpha=0.85,
            label='Before flip')
axes[1].bar(x8 + w8/2, o_8_after,  w8, color='coral', alpha=0.85,
            label='After flip')
for i in range(len(sentence_8)):
    delta = o_8_after[i] - o_8_before[i]
    if abs(delta) > 1e-4:
        ypos = max(o_8_after[i], o_8_before[i])
        axes[1].annotate(f'd={delta:+.4f}',
                         xy=(x8[i] + w8/2, ypos),
                         xytext=(0, 8), textcoords='offset points',
                         ha='center', fontsize=7, color='red')
axes[1].set_xticks(x8)
axes[1].set_xticklabels(sentence_8, rotation=45, ha='right')
axes[1].set_ylabel('$o_i = \\sum_j P_{attn,ij} \\cdot m_j$')
axes[1].set_title('Attention Output per Query\n'
                  'Weighted magnetisation = how each query\n'
                  'responds to the spin flip perturbation', fontsize=9)
axes[1].legend()
axes[1].axhline(0, color='black', lw=0.5)
axes[1].grid(True, alpha=0.3, axis='y')

fig7.suptitle('Magnetisation & Attention Output: Before vs After Spin Flip\n'
              f'8 tokens, q={q_8}, beta={beta_8:.3f}  |  '
              f'Flip: "{sentence_8[CAT_IDX_8]}" spin {FLIP_SPIN}',
              fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
print("  [Figure 7 saved: Magnetisation & output before/after flip]")


# ── Figure 8: beta sweep on 8-token showing attention sharpness ──────────────
# Use 8-token system's J and critical beta
J_8_sym     = 0.5 * (J_8 + J_8.T)
eigvals_8   = np.linalg.eigvalsh(J_8_sym)
lambda_max8 = eigvals_8.max()
beta_c_8    = 1.0 / (q_8 * lambda_max8) if lambda_max8 > 0 else np.inf

fig8, axes = plt.subplots(1, 4, figsize=(20, 4.5))

beta_vals = [0.05, beta_8, beta_c_8, beta_c_8 * 4.0]
labels    = ['Paramagnetic\n(very hot)',
             f'Attention equiv.\nbeta={beta_8:.3f}',
             f'Critical\nbeta_c={beta_c_8:.3f}',
             'Ferromagnetic\n(very cold)']

for ax, bv, lab in zip(axes, beta_vals, labels):
    P_bv = boltzmann_attention(J_8, bv, mu, q_8)
    sns.heatmap(P_bv, cmap='Blues', ax=ax,
                xticklabels=sentence_8, yticklabels=sentence_8,
                linewidths=0.1, vmin=0, vmax=1)
    for b in lattice_boundaries:
        ax.axhline(b, color='red', lw=1.5)
        ax.axvline(b, color='red', lw=1.5)
    H = -np.sum(P_bv * np.log(P_bv + 1e-30)) / len(sentence_8)
    ax.set_title(f'{lab}\nH = {H:.2f}', fontsize=9)
    ax.set_xlabel('Key'); ax.set_ylabel('Query')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

fig8.suptitle('Temperature Controls Attention Sharpness (8 tokens)\n'
              'beta->0 (hot, paramagnetic): uniform attention  |  '
              'beta->inf (cold, ferromagnetic): peaked attention',
              fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
print("  [Figure 8 saved: beta sweep on 8-token system]")


# =============================================================================
# MAPPING TABLE
# =============================================================================
print()
print("=" * 70)
print("COMPLETE MAPPING TABLE")
print("=" * 70)
rows = [
    ("Coupling J_ij = qi.kj/sqrt(d_k)",          "QK similarity score"),
    ("Boltzmann exp(bmqJ)/Sum exp(bmqJ)",         "Softmax attention weight"),
    ("  with b*m*q = 1",                          "  exact equivalence"),
    ("Equil. magnetisation m_j (global MF)",      "Value V_j"),
    ("Pair energy E_ij = -mu*q*J_ij*m_j",        "Neg log-attention (stability)"),
    ("Z_i = Sum_j exp(bmqJ_ij)",                  "Softmax denominator"),
    ("P_spin = sigma(2*bmqJ_ij)",                 "Microscopic spin probability"),
    ("<M_i> = Sum_j P_attn_ij * m_j",            "Attention output"),
    ("beta->inf (T->0, ferromagnetic)",           "Sharp / peaked attention"),
    ("beta->0  (T->inf, paramagnetic)",           "Uniform / flat attention"),
    ("beta_c = 1/(q*lambda_max)",                 "Phase transition = sharpening"),
    ("B=0: spontaneous symmetry breaking",        "Self-attention (no external bias)"),
    ("B!=0: explicit magnetisation",              "Biased attention / external signal"),
    ("Coord. q = neighbours in corr. length",     "Effective attention neighbourhood"),
    ("M = (1/N)Sum|m_i| (order parameter)",       "Avg value magnitude (per dipole)"),
    ("Spin flip in embedding",                    "Token perturbation -> attn shift"),
    ("Domain formation",                          "Attention clusters"),
]
print(f"  {'Mean-Field Ising':<48}  {'Attention Mechanism'}")
print(f"  {'-'*48}  {'-'*35}")
for left, right in rows:
    print(f"  {left:<48}  {right}")
print()
print("Key insight: energy minimisation = magnetic alignment = domain")
print("formation = maximum attention. The Boltzmann distribution over")
print("pair energies IS the softmax distribution over QK similarities.")