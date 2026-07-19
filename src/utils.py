"""
utils.py
Fungsi utilitas umum untuk proyek.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def set_plot_style():
    """Set style konsisten untuk semua plot"""
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.figsize": (10, 6),
        "font.size": 12,
    })


def save_fig(fig, filename, folder="reports"):
    """Simpan figure ke folder reports"""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"Figure disimpan: {path}")
    return path


def print_class_distribution(y, label="Target"):
    counts = pd.Series(y).value_counts().sort_index()
    print(f"\n{label} Distribution:")
    for idx, val in counts.items():
        pct = val / len(y) * 100
        print(f"  Class {idx}: {val:,} ({pct:.1f}%)")


def cramers_v(x, y):
    """Hitung Cramer's V correlation antara dua variabel kategorik"""
    from scipy.stats import chi2_contingency
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom == 0:
        return 0
    return np.sqrt(phi2corr / denom)
