"""
Mean-Field Ising → Attention
=============================
Each token = 8 spins at the corners of a unit cube.
q = 5 is the mean-field coordination number (each spin feels ~5 neighbours
on average via the mean field — no explicit wiring needed at this stage).

Self-consistent equation per (i,j) pair:
    m_ij <- tanh(beta * mu * q * J_ij * m_ij)

Effective field at convergence:
    h_ij = beta * mu * q * J_ij * m_ij

Emergent attention (fell out of Z, not applied):
    P_attn_ij = exp(h_ij) / sum_j exp(h_ij)

Boltzmann weight uses -beta*E so lower energy = higher attention:
    E_ij = -mu * q * J_ij * m_ij
    A_ij = exp(-beta * E_ij) / sum_j exp(-beta * E_ij)
         = exp(h_ij) / sum_j exp(h_ij)   [same as P_attn]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

np.random.seed(42)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['font.size']        = 10

# ── Constants ──────────────────────────────────────────────────────────────────
mu       = 1.0
beta     = 1.0
q        = 5        # mean-field coordination number
tol      = 1e-8
max_iter = 10000
d_model  = 8        # 8 spins = 8 corners of a unit cube
d_k      = 8

# ── Unit cube corner coordinates (8 corners) ──────────────────────────────────
CUBE_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=float)

# Cube edges for drawing
CUBE_EDGES = [
    (0,1),(2,3),(4,5),(6,7),   # x-direction
    (0,2),(1,3),(4,6),(5,7),   # y-direction
    (0,4),(1,5),(2,6),(3,7),   # z-direction
]

# ── Sentence: 12 tokens, each is a ±1 spin config at 8 cube corners ───────────
sentence = [
    'the',  'old',
    'cat',  'sat',
    'on',   'warm',
    'mat',  'today',
]
N = len(sentence)
CAT_IDX = sentence.index('cat')

np.random.seed(7)
X = np.array([
    np.random.choice([-1.0, 1.0], size=d_model)
    for _ in sentence
])

# ── Projection matrices ────────────────────────────────────────────────────────
np.random.seed(42)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5


def compute_J(X):
    Q = X @ W_Q
    K = X @ W_K
    return Q @ K.T / np.sqrt(d_k)


# ── Self-consistent mean-field ─────────────────────────────────────────────────

def solve_mean_field(J, beta, mu, q, tol, max_iter):
    """
    Solve m_ij <- tanh(beta*mu*q*J_ij*m_ij) per (i,j) pair.
    Seeded at sign(J)*0.5 for correct bifurcation side.
    Returns converged m (N,N).
    """
    m          = np.sign(J) * 0.5
    m[m == 0]  = 0.5
    scale      = beta * mu * q
    for _ in range(max_iter):
        m_new = np.tanh(scale * J * m)
        if np.abs(m_new - m).max() < tol:
            return m_new
        m = m_new
    return m


def compute_attention(J, beta, mu, q, tol, max_iter):
    """
    Returns P_attn (N,N) — emergent attention from partition function.
    Lower energy (more stable) pair gets higher Boltzmann weight.
    """
    m      = solve_mean_field(J, beta, mu, q, tol, max_iter)
    h      = beta * mu * q * J * m          # effective field at convergence
    h_s    = h - h.max(axis=1, keepdims=True)
    P_attn = np.exp(h_s) / np.exp(h_s).sum(axis=1, keepdims=True)
    return P_attn, m


# ── Compute before and after spin flip ────────────────────────────────────────
J_before              = compute_J(X)
P_before, m_before    = compute_attention(J_before, beta, mu, q, tol, max_iter)

# Flip spin at corner 2 of 'cat'
X_after               = X.copy()
FLIP_SPIN             = 2
X_after[CAT_IDX, FLIP_SPIN] *= -1
J_after               = compute_J(X_after)
P_after, m_after      = compute_attention(J_after, beta, mu, q, tol, max_iter)

dP = P_after - P_before
max_idx = np.unravel_index(np.abs(dP).argmax(), dP.shape)

print(f"Spin flip: '{sentence[CAT_IDX]}' corner {FLIP_SPIN}  "
      f"{int(X[CAT_IDX, FLIP_SPIN]):+d} -> {int(X_after[CAT_IDX, FLIP_SPIN]):+d}")
print(f"Most changed cell: ({sentence[max_idx[0]]} -> {sentence[max_idx[1]]})  "
      f"dP = {dP[max_idx]:+.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — 8 cubes joined in a 4×2 grid in a single 3D axis
# Cubes share faces — adjacent tokens are spatially connected
# Layout: 4 columns (lattice groups) × 2 rows
#   col 0: the, old      col 1: cat, sat
#   col 2: on, warm      col 3: mat, today
# ═══════════════════════════════════════════════════════════════════════════════

# Grid positions for each token: (col, row) in cube units
# gap = 0 means cubes share faces
GRID = [
    (0, 1),  # the
    (0, 0),  # old
    (1, 1),  # cat
    (1, 0),  # sat
    (2, 1),  # on
    (2, 0),  # warm
    (3, 1),  # mat
    (3, 0),  # today
]

fig1 = plt.figure(figsize=(16, 8))
ax3d = fig1.add_subplot(111, projection='3d')

for idx, (token, (gx, gy)) in enumerate(zip(sentence, GRID)):
    spins  = X[idx]
    offset = np.array([gx, gy, 0], dtype=float)  # cube origin in grid
    is_cat = (idx == CAT_IDX)

    # draw cube edges — shifted by offset
    for e0, e1 in CUBE_EDGES:
        p0 = CUBE_CORNERS[e0] + offset
        p1 = CUBE_CORNERS[e1] + offset
        lc  = '#cc0000' if is_cat else '#888888'
        lw  = 2.0       if is_cat else 0.8
        ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                  color=lc, lw=lw, alpha=0.8)

    # draw spins at corners
    for k, (corner, spin) in enumerate(zip(CUBE_CORNERS, spins)):
        pos      = corner + offset
        is_flip  = (is_cat and k == FLIP_SPIN)
        color    = '#e74c3c' if is_flip else ('#2196F3' if spin > 0 else '#FF9800')
        size     = 200 if is_flip else 80
        marker   = '*'  if is_flip else ('^' if spin > 0 else 'v')
        ax3d.scatter(*pos, c=color, s=size, marker=marker,
                     zorder=5, depthshade=False)

    # token label below cube
    label_color = '#cc0000' if is_cat else '#333333'
    label_weight = 'bold' if is_cat else 'normal'
    ax3d.text(gx + 0.5, gy + 0.5, -0.35,
              f'"{token}"\n↑{int((spins==1).sum())} ↓{int((spins==-1).sum())}',
              ha='center', va='top', fontsize=8,
              color=label_color, fontweight=label_weight)

# draw lattice group boundaries as vertical planes (dashed lines at x=1,2,3)
for bx in [1.0, 2.0, 3.0]:
    for y in [0.0, 1.0]:
        ax3d.plot([bx, bx], [y, y+1], [0, 0],
                  color='darkred', lw=2, ls='--', alpha=0.5)
        ax3d.plot([bx, bx], [y, y+1], [1, 1],
                  color='darkred', lw=2, ls='--', alpha=0.5)

# lattice group labels
group_names_3d = ['Lattice 1\n(the, old)',
                  'Lattice 2\n(cat, sat)',
                  'Lattice 3\n(on, warm)',
                  'Lattice 4\n(mat, today)']
for g, name in enumerate(group_names_3d):
    ax3d.text(g + 0.5, 2.0, 1.1, name,
              ha='center', va='bottom', fontsize=8,
              color='darkred', fontweight='bold')

ax3d.set_xlim(-0.2, 4.2)
ax3d.set_ylim(-0.5, 2.5)
ax3d.set_zlim(-0.5, 1.5)
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
ax3d.set_box_aspect([4, 2, 1])
ax3d.view_init(elev=20, azim=-60)

# legend
up_patch   = mpatches.Patch(color='#2196F3', label='Spin +1 (up) ▲')
down_patch = mpatches.Patch(color='#FF9800', label='Spin −1 (down) ▼')
flip_patch = mpatches.Patch(color='#e74c3c',
                             label=f'Flip target: "cat" corner {FLIP_SPIN} ★')
ax3d.legend(handles=[up_patch, down_patch, flip_patch],
            loc='upper left', fontsize=9)

ax3d.set_title(
    '8 Tokens as Joined Spin Cubes  —  4 lattice groups × 2 tokens each\n'
    'Each cube = 8 spins (±1) at corners  |  Shared faces = neighbouring tokens\n'
    f'q={q} mean-field coordination  |  "cat" in red  |  ★ = spin to be flipped',
    fontsize=11, fontweight='bold', pad=15)

plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — P_attn heatmap before flip, cat row highlighted
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(P_before, cmap='Blues', ax=ax,
            xticklabels=sentence, yticklabels=sentence,
            linewidths=0.3, vmin=0, vmax=P_before.max(),
            annot=True, fmt='.3f', annot_kws={'size': 8})

# highlight cat row and column
for spine in ax.spines.values():
    spine.set_visible(True)

# draw red rectangle around cat's row
ax.add_patch(plt.Rectangle(
    (0, CAT_IDX), N, 1,
    fill=False, edgecolor='red', lw=2.5, zorder=5))
ax.add_patch(plt.Rectangle(
    (CAT_IDX, 0), 1, N,
    fill=False, edgecolor='red', lw=1.5, ls='--', zorder=5, alpha=0.6))

# lattice group boundaries
for boundary in [2, 4, 6]:
    ax.axhline(boundary, color='darkred', lw=2, ls='-')
    ax.axvline(boundary, color='darkred', lw=2, ls='-')

# lattice group labels
group_names = ['Lattice 1', 'Lattice 2', 'Lattice 3', 'Lattice 4']
for g in range(4):
    mid = g * 2 + 1.0
    ax.text(mid, -0.8, group_names[g],
            ha='center', va='bottom', fontsize=8,
            color='darkred', fontweight='bold',
            transform=ax.get_xaxis_transform())

ax.set_title(
    'Emergent Attention $P_{attn}$  —  Before Spin Flip\n'
    '$P_{attn,ij} = e^{h_{ij}} / \\sum_j e^{h_{ij}}$  where  '
    '$h_{ij} = \\beta \\mu q J_{ij} m_{ij}$  (converged)\n'
    'Red box = "cat" row (query) and column (key)  |  '
    'Dark lines = lattice group boundaries',
    fontsize=10)
ax.set_xlabel('Key token  $k_j$  (which token is being attended to)')
ax.set_ylabel('Query token  $q_i$  (which token is asking)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — ΔP_attn after flipping one spin in cat
# ═══════════════════════════════════════════════════════════════════════════════

fig3, ax = plt.subplots(figsize=(10, 8))

vmax_d = np.abs(dP).max()
sns.heatmap(dP, cmap='coolwarm', center=0, ax=ax,
            xticklabels=sentence, yticklabels=sentence,
            linewidths=0.3, vmin=-vmax_d, vmax=vmax_d,
            annot=False)

# lattice boundaries
for boundary in [2, 4, 6]:
    ax.axhline(boundary, color='black', lw=2)
    ax.axvline(boundary, color='black', lw=2)

# highlight cat row
ax.add_patch(plt.Rectangle(
    (0, CAT_IDX), N, 1,
    fill=False, edgecolor='red', lw=2.5, zorder=5))

# circle most changed cell
r_l, c_l = max_idx
ax.add_patch(mpatches.Circle(
    (c_l + 0.5, r_l + 0.5), 0.48,
    fill=False, edgecolor='lime', lw=3, zorder=6))
ax.text(c_l + 0.5, r_l - 0.55,
        f'most changed\n({sentence[r_l]}→{sentence[c_l]})\ndP={dP[max_idx]:+.4f}',
        ha='center', va='bottom', fontsize=8,
        color='lime', fontweight='bold')

# group labels
for g in range(4):
    mid = g * 2 + 1.0
    ax.text(mid, -0.8, group_names[g],
            ha='center', va='bottom', fontsize=8,
            color='black', fontweight='bold',
            transform=ax.get_xaxis_transform())

ax.set_title(
    f'$\\Delta P_{{attn}}$ = After − Before  '
    f'(flip: "cat" corner {FLIP_SPIN}  '
    f'{int(X[CAT_IDX, FLIP_SPIN]):+d}→{int(X_after[CAT_IDX, FLIP_SPIN]):+d})\n'
    'Red = attention increased  |  Blue = attention decreased\n'
    'One spin flip in "cat" propagates through J → m → h → P_attn across the whole lattice',
    fontsize=10)
ax.set_xlabel('Key token  $k_j$')
ax.set_ylabel('Query token  $q_i$')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()