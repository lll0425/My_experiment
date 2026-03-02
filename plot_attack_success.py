"""
Plot attack success rate line chart from attack_success_rate_mean_acc.csv.

Usage (from repo root or NeurIPS24_FDCR_Code):
  python plot_attack_success.py --csv_path path/to/attack_success_rate_mean_acc.csv
"""

import argparse
import os
import csv
import re

import matplotlib.pyplot as plt


def load_values(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    if len(rows) < 2:
        raise ValueError("CSV should have header row and one data row.")
    header = rows[0]
    values = [float(x) for x in rows[1]]
    # epoch labels might be like epoch_0, epoch_1...
    epochs = list(range(len(values)))
    return epochs, values


def infer_non_iid(csv_path: str) -> str:
    data_dir = os.path.dirname(csv_path)
    cfg_path = os.path.join(data_dir, "cfg.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"beta:\s*([0-9]+(?:\.[0-9]+)?)", line)
                if m:
                    return m.group(1)

    parts = os.path.normpath(data_dir).split(os.sep)
    for p in reversed(parts):
        if p.startswith("beta"):
            tail = p.replace("beta", "", 1)
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", tail)
            if m:
                return m.group(1)
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", p):
            return p
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Plot attack success rate line chart.")
    ap.add_argument("--csv_path", required=True, help="attack_success_rate_mean_acc.csv")
    ap.add_argument("--out_path", default=None, help="Optional output image path")
    args = ap.parse_args()

    epochs, values = load_values(args.csv_path)
    out_path = args.out_path or os.path.join(os.path.dirname(args.csv_path), "attack_success_rate.png")
    non_iid = infer_non_iid(args.csv_path)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, values, color="#c00000", linewidth=2)
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Attack success rate (%)")
    plt.title(f"Attack Sucess Rate (non-iid={non_iid})")
    # mark highest (exclude first epoch)
    if len(values) > 1:
        max_idx = 1 + max(range(len(values) - 1), key=lambda i: values[i + 1])
        max_val = values[max_idx]
        plt.scatter([max_idx], [max_val], color="#1f77b4", zorder=3)
        plt.annotate(
            f"max={max_val:.2f}% @ epoch_{max_idx}",
            (max_idx, max_val),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=8,
            color="#1f77b4",
        )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
