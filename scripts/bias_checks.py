"""
bias_checks.py
==============
Runs all bias and quality checks on the DeepLoc SCL and FLIP Meltome datasets.
Produces summary tables and saves results to bias_check_results.csv

Usage:
    python bias_checks.py

Requirements:
    pip install pandas numpy scipy matplotlib seaborn

Datasets expected in the same folder (or update paths below):
    - subdataset_scl_100.csv   (or full: balanced.csv from FLIP SCL split)
    - subdataset_meltome_100.csv (or full: human_cell.csv from FLIP Meltome split)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from math import log2

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Swap to the full datasets for complete analysis:
SCL_PATH   = os.path.join(SCRIPT_DIR, "deeploc_scl_full.csv")
MELT_PATH  = os.path.join(SCRIPT_DIR, "meltome_human_cell_full.csv")
OUT_CSV    = os.path.join(SCRIPT_DIR, "bias_check_results.csv")

UNIPROT_FREQ = {
    'A':0.0825,'R':0.0553,'N':0.0406,'D':0.0545,'C':0.0137,'Q':0.0393,'E':0.0675,
    'G':0.0707,'H':0.0227,'I':0.0595,'L':0.0966,'K':0.0584,'M':0.0242,'F':0.0386,
    'P':0.0470,'S':0.0657,'T':0.0534,'W':0.0108,'Y':0.0292,'V':0.0687
}
BAD_AA = set('XBZUO')

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading datasets...")
scl  = pd.read_csv(SCL_PATH)
melt = pd.read_csv(MELT_PATH)
scl['seq_len']  = scl['sequence'].str.len()
melt['seq_len'] = melt['sequence'].str.len()
print(f"  SCL:     {len(scl):,} sequences")
print(f"  Meltome: {len(melt):,} sequences")

results = []  # collect rows for output CSV

# ════════════════════════════════════════════════════════════════════════════
# 1. CLASS BALANCE (SCL)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. SCL CLASS BALANCE")
print("="*60)

vc = scl['target'].value_counts()
probs = vc / vc.sum()
H = -sum(p * log2(p) for p in probs if p > 0)
H_max = log2(len(vc))
print(f"\n{'Class':<25} {'Count':>6}  {'%':>6}")
print("-" * 42)
for cls, cnt in vc.items():
    pct = cnt / len(scl) * 100
    print(f"  {cls:<23} {cnt:>6,}  {pct:>5.1f}%")
print(f"\n  Shannon Entropy: H = {H:.4f}  (max = {H_max:.4f})")
print(f"  Balance score:   {H/H_max*100:.1f}%  (100% = perfectly balanced)")

for cls, cnt in vc.items():
    results.append({'dataset':'SCL','check':'class_balance','item':cls,
                    'value':cnt,'pct':round(cnt/len(scl)*100,2)})
results.append({'dataset':'SCL','check':'shannon_entropy','item':'H',
                'value':round(H,4),'pct':round(H/H_max*100,1)})

# ════════════════════════════════════════════════════════════════════════════
# 2. Tm DISTRIBUTION (MELTOME)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. MELTOME Tm DISTRIBUTION")
print("="*60)

tm = melt['target']
print(f"\n  N:        {len(tm):,}")
print(f"  Mean:     {tm.mean():.2f} °C")
print(f"  Median:   {tm.median():.2f} °C")
print(f"  Std:      {tm.std():.2f} °C")
print(f"  Min:      {tm.min():.2f} °C")
print(f"  Max:      {tm.max():.2f} °C")
print(f"  Skewness: {tm.skew():.4f}")
print(f"  Kurtosis: {tm.kurt():.4f}")

# Quintile breakdown
melt['quintile'] = pd.qcut(tm, q=5, labels=['Q1 (cold)','Q2','Q3','Q4','Q5 (hot)'])
print(f"\n  Tm Quintile ranges:")
for q, grp in melt.groupby('quintile', observed=True):
    print(f"    {q}: {grp['target'].min():.1f} – {grp['target'].max():.1f} °C  (n={len(grp)})")
    results.append({'dataset':'Meltome','check':'tm_quintile','item':str(q),
                    'value':len(grp),'pct':round(len(grp)/len(melt)*100,1)})

# ════════════════════════════════════════════════════════════════════════════
# 3. SEQUENCE LENGTH BIAS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. SEQUENCE LENGTH ANALYSIS")
print("="*60)

for name, df, label_col in [("SCL", scl, "target"), ("Meltome", melt, "target")]:
    print(f"\n  {name}:")
    print(f"    Mean:   {df['seq_len'].mean():.0f} aa")
    print(f"    Median: {df['seq_len'].median():.0f} aa")
    print(f"    Std:    {df['seq_len'].std():.0f} aa")
    print(f"    Min:    {df['seq_len'].min()} aa")
    print(f"    Max:    {df['seq_len'].max():,} aa")

# SCL: length per class
print("\n  SCL — Mean length per class:")
print(f"  {'Class':<25} {'Mean':>6}  {'Median':>7}  {'Std':>6}")
print("  " + "-"*50)
class_len = scl.groupby('target')['seq_len'].agg(['mean','median','std']).sort_values('mean')
for cls, row in class_len.iterrows():
    print(f"  {cls:<25} {row['mean']:>6.0f}  {row['median']:>7.0f}  {row['std']:>6.0f}")
    results.append({'dataset':'SCL','check':'length_per_class','item':cls,
                    'value':round(row['mean'],1),'pct':round(row['median'],1)})

# Meltome: length vs Tm correlation
r_val, p_val = stats.spearmanr(melt['seq_len'], melt['target'])
print(f"\n  Meltome — Length vs Tm:")
print(f"    Spearman R = {r_val:.4f}  (p = {p_val:.2e})")
if abs(r_val) < 0.1:
    interp = "negligible bias"
elif abs(r_val) < 0.3:
    interp = "weak bias"
elif abs(r_val) < 0.5:
    interp = "moderate bias"
else:
    interp = "strong bias — flag for review"
print(f"    Interpretation: {interp}")
results.append({'dataset':'Meltome','check':'length_vs_tm_spearman','item':'R',
                'value':round(r_val,4),'pct':round(p_val,6)})

# ════════════════════════════════════════════════════════════════════════════
# 4. AMINO ACID FREQUENCY BIAS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. AMINO ACID COMPOSITION")
print("="*60)

def aa_freq(seqs):
    counter = Counter()
    total = 0
    for s in seqs:
        for aa in s:
            if aa in UNIPROT_FREQ:
                counter[aa] += 1
                total += 1
    return {aa: counter.get(aa,0)/total for aa in sorted(UNIPROT_FREQ)}, total

scl_freq, scl_total   = aa_freq(scl['sequence'])
melt_freq, melt_total = aa_freq(melt['sequence'])

print(f"\n  {'AA':<4} {'UniProt':>8}  {'SCL':>8}  {'SCL Δ':>8}  {'Meltome':>8}  {'Melt Δ':>8}")
print("  " + "-"*56)
for aa in sorted(UNIPROT_FREQ):
    uni  = UNIPROT_FREQ[aa]
    s    = scl_freq[aa]
    m    = melt_freq[aa]
    ds   = s - uni
    dm   = m - uni
    flag_s = " ◄" if abs(ds) > 0.01 else ""
    flag_m = " ◄" if abs(dm) > 0.01 else ""
    print(f"  {aa:<4} {uni:>8.4f}  {s:>8.4f}  {ds:>+8.4f}{flag_s}  {m:>8.4f}  {dm:>+8.4f}{flag_m}")
    results.append({'dataset':'both','check':'aa_freq','item':aa,
                    'value':round(s,4),'pct':round(m,4)})

print("\n  ◄ = deviation > 0.01 from UniProt baseline")

# ════════════════════════════════════════════════════════════════════════════
# 5. QUALITY CHECKS (unknown residues, extreme lengths)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. SEQUENCE QUALITY CHECKS")
print("="*60)

for name, df in [("SCL", scl), ("Meltome", melt)]:
    n_bad  = sum(1 for s in df['sequence'] if any(c in BAD_AA for c in s))
    n_x    = sum(s.count('X') for s in df['sequence'])
    n_short = (df['seq_len'] < 20).sum()
    n_long  = (df['seq_len'] > 2000).sum()
    print(f"\n  {name} (n={len(df):,}):")
    print(f"    Sequences with unknown residues (X/B/Z/U/O): {n_bad:,}  ({n_bad/len(df)*100:.2f}%)")
    print(f"    Total 'X' residues:   {n_x:,}")
    print(f"    Very short (<20 aa):  {n_short:,}")
    print(f"    Very long  (>2000 aa): {n_long:,}  ({n_long/len(df)*100:.1f}%)")
    results.append({'dataset':name,'check':'quality_unknown_aa','item':'n_seqs_with_bad_aa',
                    'value':n_bad,'pct':round(n_bad/len(df)*100,3)})
    results.append({'dataset':name,'check':'quality_very_long','item':'>2000aa',
                    'value':n_long,'pct':round(n_long/len(df)*100,1)})

# ════════════════════════════════════════════════════════════════════════════
# 6. TRAIN / TEST SPLIT COMPARISON
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. TRAIN / TEST SPLIT COMPARISON")
print("="*60)

for name, df in [("SCL", scl), ("Meltome", melt)]:
    has_set = 'set' in df.columns
    if not has_set:
        print(f"\n  {name}: no 'set' column (sub-dataset only — run on full split CSV for this check)")
        continue
    train = df[df['set']=='train']
    test  = df[df['set']=='test']
    print(f"\n  {name}:")
    print(f"    Train: {len(train):,}  |  Test: {len(test):,}  |  Ratio: {len(test)/len(df)*100:.1f}% test")
    for split_name, split_df in [("Train", train), ("Test", test)]:
        print(f"    {split_name} lengths — mean:{split_df['seq_len'].mean():.0f}  "
              f"median:{split_df['seq_len'].median():.0f}  std:{split_df['seq_len'].std():.0f}")
    # KS test for length distribution difference
    ks_stat, ks_p = stats.ks_2samp(train['seq_len'], test['seq_len'])
    print(f"    KS test (lengths train vs test): stat={ks_stat:.4f}, p={ks_p:.4f}", end="")
    print("  ← distributions differ" if ks_p < 0.05 else "  ✓ distributions match")
    results.append({'dataset':name,'check':'train_test_ks_length','item':'KS_stat',
                    'value':round(ks_stat,4),'pct':round(ks_p,4)})

# ════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════
df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)
print(f"\n{'='*60}")
print(f"All results saved to: {OUT_CSV}")
print(f"{'='*60}")
