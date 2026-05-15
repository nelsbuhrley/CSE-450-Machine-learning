import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})

with open("/sessions/clever-hopeful-keller/mnt/outputs/analysis/nels_full_loo.json") as fh:
    R = json.load(fh)

baseline_v = R['BASELINE__seed42']['v']

# === Figure 1: LOO bar chart (single feature drop) ===
loo_features = []
for k in R:
    if k.startswith('drop_') and '__seed42' in k:
        feat = k.replace('drop_', '').replace('__seed42','')
        delta = R[k]['v'] - baseline_v
        loo_features.append((feat, R[k]['v'], delta, R[k]['tp'], R[k]['fp']))

loo_features.sort(key=lambda x: x[2])  # sort by delta
fig, ax = plt.subplots(figsize=(11, 8))
names = [f[0] for f in loo_features]
deltas = [f[2] for f in loo_features]
colors = ['#2E7D32' if d > 30 else '#90A4AE' if d > -30 else '#C62828' for d in deltas]
bars = ax.barh(names, deltas, color=colors, edgecolor='white')
ax.axvline(0, color='black', lw=1)
ax.set_xlabel(f"Δ$ vs baseline (baseline = ${baseline_v:.2f})")
ax.set_title("Nels's stacking model: leave-one-out feature impact (full data, seed=42)\nGreen = removing this feature HELPS  |  Red = HURTS  |  Gray = wash",
             fontweight='bold')
for b, (_, v, d, tp, fp) in zip(bars, loo_features):
    label = f"  ${v:.0f}  (TP={tp},FP={fp})  Δ{d:+.0f}"
    ax.text(d if d > 0 else 0, b.get_y()+b.get_height()/2, label, va='center', fontsize=8)
plt.tight_layout()
out = "/sessions/clever-hopeful-keller/mnt/CSE-450-Machine-learning/nels_b/nels_loo_full.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")

# === Figure 2: variant comparison across seeds ===
def get_seeds(name):
    seeds = {}
    for s in [42, 0, 7]:
        k = f"{name}__seed{s}"
        if k in R: seeds[s] = R[k]['v']
    return seeds

variants = {
    'baseline (Caleb method)': get_seeds('BASELINE'),
    'no demographics': get_seeds('no_demographics'),
    'macro_only (5 macros + poutcome)': get_seeds('macro_only'),
    'only 5 macros (no poutcome)': get_seeds('only5_macros'),
}

fig2, axes = plt.subplots(1, 2, figsize=(15, 5.5))
seeds = [42, 0, 7]
x = np.arange(len(seeds))
w = 0.21
colors_v = ['#1565C0','#7B1FA2','#2E7D32','#F57C00']
for i,(name, sd) in enumerate(variants.items()):
    vals = [sd.get(s, 0) for s in seeds]
    axes[0].bar(x + (i-1.5)*w, vals, w, label=name, color=colors_v[i])
axes[0].axhline(650, color='#2E7D32', ls='--', lw=1.2, alpha=0.7, label='Blue threshold ($650)')
axes[0].set_xticks(x); axes[0].set_xticklabels([f'seed={s}' for s in seeds])
axes[0].set_ylabel("Value of calls ($)")
axes[0].set_title("Per-seed performance", fontweight='bold')
axes[0].legend(loc='lower right', fontsize=8)
axes[0].set_ylim(0, 950)

# Mean comparison
means = {n: np.mean(list(sd.values())) for n,sd in variants.items()}
axes[1].bar(range(len(means)), list(means.values()), color=colors_v)
axes[1].axhline(650, color='#2E7D32', ls='--', lw=1.2, alpha=0.7, label='Blue threshold ($650)')
axes[1].set_xticks(range(len(means)))
axes[1].set_xticklabels([n[:25] for n in means.keys()], rotation=12, ha='right', fontsize=8)
axes[1].set_ylabel("Mean value of calls ($)")
axes[1].set_title("Mean across 3 seeds", fontweight='bold')
axes[1].set_ylim(0, 950)
for i,(n,v) in enumerate(means.items()):
    axes[1].text(i, v+10, f"${v:.0f}", ha='center', fontsize=10, fontweight='bold')
base_mean = means['baseline (Caleb method)']
for i,(n,v) in enumerate(means.items()):
    if i > 0:
        axes[1].text(i, 30, f"Δ {v-base_mean:+.0f}", ha='center', fontsize=9, color='white', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, pad=2))
axes[1].legend(loc='upper left', fontsize=9)
plt.tight_layout()
out2 = "/sessions/clever-hopeful-keller/mnt/CSE-450-Machine-learning/nels_b/nels_variant_comparison.png"
plt.savefig(out2, dpi=140, bbox_inches='tight', facecolor='white')
print(f"Saved: {out2}")
