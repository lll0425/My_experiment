"""
Plot attack success rate for two CSVs in one chart, marking max excluding epoch 0.

Usage (from NeurIPS24_FDCR_Code):
  python plot_attack_two.py --csv1 path/to/attack_success_rate_mean_acc.csv \
                            --csv2 path/to/attack_success_rate_mean_acc.csv
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_values(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if row]
    if len(rows) < 2:
        raise ValueError(f"CSV should have header + one row: {csv_path}")
    values = [float(x) for x in rows[1]]
    epochs = list(range(len(values)))
    return epochs, values


def plot_compare(csv1: str, csv2: str, out_path: str, label1: str, label2: str):
    e1, v1 = load_values(csv1)
    e2, v2 = load_values(csv2)

    plt.figure(figsize=(8, 4))
    plt.plot(e1, v1, color="#c00000", linewidth=2, label=label1)
    plt.plot(e2, v2, color="#1f77b4", linewidth=2, label=label2)
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Attack success rate (%)")
    plt.title("Attack Success Rate Comparison (non-iid 0.3)")

    # mark max (exclude epoch 0) for each series
    if len(v1) > 1:
        max_idx = 1 + max(range(len(v1) - 1), key=lambda i: v1[i + 1])
        max_val = v1[max_idx]
        plt.scatter([max_idx], [max_val], color="#c00000", zorder=3)
        plt.annotate(
            f"max={max_val:.2f}% @ {max_idx}",
            (max_idx, max_val),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=8,
            color="#c00000",
        )
    if len(v2) > 1:
        max_idx = 1 + max(range(len(v2) - 1), key=lambda i: v2[i + 1])
        max_val = v2[max_idx]
        plt.scatter([max_idx], [max_val], color="#1f77b4", zorder=3)
        plt.annotate(
            f"max={max_val:.2f}% @ {max_idx}",
            (max_idx, max_val),
            textcoords="offset points",
            xytext=(6, -12),
            ha="left",
            fontsize=8,
            color="#1f77b4",
        )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot attack success rate for two CSVs.")
    ap.add_argument("--csv1", required=True, help="Path to first attack_success_rate_mean_acc.csv")
    ap.add_argument("--csv2", required=True, help="Path to second attack_success_rate_mean_acc.csv")
    ap.add_argument("--out_path", default=None, help="Output image path")
    ap.add_argument("--label1", default="baseline", help="Legend label for csv1 (red)")
    ap.add_argument("--label2", default="slice", help="Legend label for csv2 (blue)")
    args = ap.parse_args()

    out_path = args.out_path or os.path.join(os.path.dirname(args.csv1), "attack_success_rate_compare.png")
    plot_compare(args.csv1, args.csv2, out_path, args.label1, args.label2)


if __name__ == "__main__":
    main()
