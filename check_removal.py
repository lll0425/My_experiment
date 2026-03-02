"""
python check_removal.py --csv_path data/label_skew/base_backdoor/0.3/fl_fashionmnist/100.0/Ours2/fedfish/beta100.0_Ours2/aggregation_weight.csv --plot_simple                               


Check per-epoch client removal correctness.

Assumptions:
- aggregation_weight.csv has one row per epoch, 10 clients.
- Clients 0-6 are benign, 7-9 are malicious.
- Zero weight => client removed。非零 => 被保留。

Outputs:
- Console summary: for每個 epoch，顯示誤殺數(benign 被清零)與漏殺數(malicious 非零)。
- 可選將結果寫成 CSV。

Usage (from repo root):
  python check_removal.py --csv_path path/to/aggregation_weight.csv --out_csv stats.csv
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_weights(csv_path: str) -> np.ndarray:
    rows: List[List[float]] = []
    maxlen = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 跳過像 "47:" 的索引前綴
            if ":" in line:
                line = line.split(":", 1)[1]
            nums = [float(x) for x in line.split(",") if x]
            if not nums:
                continue
            rows.append(nums)
            maxlen = max(maxlen, len(nums))
    for r in rows:
        while len(r) < maxlen:
            r.append(0.0)
    return np.array(rows)


def check_epoch(weights: np.ndarray, benign_ids=range(7), malicious_ids=range(7, 10)) -> Tuple[np.ndarray, np.ndarray]:
    """回傳 (benign_zero_idx, malicious_kept_idx) 以 0-based client id。"""
    benign_zero = np.where(weights[list(benign_ids)] == 0)[0]
    malicious_kept = np.where(weights[list(malicious_ids)] > 0)[0]
    # map back to client ids
    benign_zero = np.array(list(benign_ids))[benign_zero]
    malicious_kept = np.array(list(malicious_ids))[malicious_kept]
    return benign_zero, malicious_kept


def main():
    ap = argparse.ArgumentParser(description="Check whether client removal is correct per epoch.")
    ap.add_argument("--csv_path", required=True, help="aggregation_weight.csv")
    ap.add_argument("--out_csv", default=None, help="Optional: write per-epoch stats to CSV")
    ap.add_argument("--plot", action="store_true", help="Save a plot of false positives/negatives per epoch")
    ap.add_argument("--plot_malicious", action="store_true", help="Only plot malicious detection rate per epoch")
    ap.add_argument("--plot_simple", action="store_true", help="Simple bar chart: how many malicious clients removed per epoch")
    args = ap.parse_args()

    w = load_weights(args.csv_path)
    benign_zero_counts = []
    malicious_kept_counts = []

    print(f"Loaded {w.shape[0]} epochs, {w.shape[1]} clients from {args.csv_path}")
    print("Epoch\tfalse_pos(誤殺良性)\tfalse_neg(漏殺惡性)\tbenign_ids_zero\tmalicious_ids_kept")
    for i, row in enumerate(w):
        bz, mk = check_epoch(row)
        benign_zero_counts.append(len(bz))
        malicious_kept_counts.append(len(mk))
        print(f"{i+1}\t{len(bz)}\t{len(mk)}\t{list(bz)}\t{list(mk)}")

    print("\nSummary:")
    print(f"Avg false positives per epoch: {np.mean(benign_zero_counts):.2f}")
    print(f"Avg false negatives per epoch: {np.mean(malicious_kept_counts):.2f}")
    print(f"Epochs with perfect detection (no FP, no FN): {(np.sum((np.array(benign_zero_counts)==0) & (np.array(malicious_kept_counts)==0)))}/{w.shape[0]}")

    if args.plot or args.plot_malicious or args.plot_simple:
        out_dir = os.path.dirname(args.csv_path)
        epochs = np.arange(1, len(benign_zero_counts) + 1)

        # Malicious detection rate plot (how many of 7-9 were removed)
        mal_weights = w[:, 7:]
        mal_detected = (mal_weights == 0).sum(axis=1)  # 0 => removed
        mal_rate = mal_detected / mal_weights.shape[1]
        if args.plot_simple:
            # simple bar chart: 0..3 removed, green if all removed
            colors = ["#c00000" if x < 3 else "#4f81bd" for x in mal_detected]
            plt.figure(figsize=(10, 4))
            plt.bar(epochs, mal_detected, color=colors)
            plt.yticks([0, 1, 2, 3])
            plt.xlabel("Epoch")
            plt.ylabel("# Malicious removed (out of 3)")
            plt.title("Malicious removal count per epoch")
            plt.grid(alpha=0.3, axis="y")
            out_path = os.path.join(out_dir, "malicious_removed_count.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved plot to {out_path}")
            if not (args.plot or args.plot_malicious):
                return

        plt.figure(figsize=(10, 4))
        plt.plot(epochs, mal_rate, color="red", linewidth=2, label="Malicious detection rate")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Epoch")
        plt.ylabel("Detection rate (0~1)")
        plt.title("Malicious detection rate per epoch (clients 7-9)")
        plt.grid(alpha=0.3)
        plt.legend()
        out_path = os.path.join(out_dir, "malicious_detection_rate.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot to {out_path}")

        if args.plot_malicious and not args.plot:
            return

        # 1) line plot of error counts
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, benign_zero_counts, label="False positives (benign removed)", color="steelblue")
        plt.plot(epochs, malicious_kept_counts, label="False negatives (malicious kept)", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Count")
        plt.title("Removal errors per epoch")
        plt.grid(alpha=0.3)
        plt.legend()
        out_path = os.path.join(out_dir, "removal_errors.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot to {out_path}")

        # 2) intuitive matrix: kept/removed + error map
        kept = (w > 0).astype(int)  # shape: (epoch, client)
        kept_map = kept.T  # clients x epochs

        # error map: 0=ok, 1=false positive (benign removed), 2=false negative (malicious kept)
        error = np.zeros_like(kept_map)
        benign_ids = np.arange(0, 7)
        malicious_ids = np.arange(7, 10)
        # benign removed -> FP
        error[benign_ids, :] = (kept_map[benign_ids, :] == 0).astype(int)
        # malicious kept -> FN (value 2)
        error[malicious_ids, :] = (kept_map[malicious_ids, :] == 1).astype(int) * 2

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        cmap_kept = ListedColormap(["#dddddd", "#4f81bd"])  # removed, kept
        im0 = axes[0].imshow(kept_map, aspect="auto", cmap=cmap_kept, interpolation="nearest")
        axes[0].set_title("Kept(blue) vs Removed(gray)")
        axes[0].set_ylabel("Client ID")
        axes[0].set_yticks(range(10))
        axes[0].set_yticklabels([str(i) for i in range(10)])

        cmap_err = ListedColormap(["#ffffff", "#f4b183", "#c00000"])  # ok, FP, FN
        im1 = axes[1].imshow(error, aspect="auto", cmap=cmap_err, interpolation="nearest")
        axes[1].set_title("Error map: FP(orange)=benign removed, FN(red)=malicious kept")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Client ID")
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([str(i) for i in range(10)])

        # x ticks as epochs (sparse)
        if kept_map.shape[1] <= 50:
            axes[1].set_xticks(range(kept_map.shape[1]))
            axes[1].set_xticklabels([str(i + 1) for i in range(kept_map.shape[1])], rotation=90, fontsize=6)
        else:
            step = max(1, kept_map.shape[1] // 10)
            axes[1].set_xticks(range(0, kept_map.shape[1], step))
            axes[1].set_xticklabels([str(i + 1) for i in range(0, kept_map.shape[1], step)])

        plt.tight_layout()
        out_path = os.path.join(out_dir, "removal_matrix.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot to {out_path}")

    if args.out_csv:
        import csv

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "false_pos", "false_neg", "benign_zero_ids", "malicious_kept_ids"])
            for i, (bz, mk) in enumerate(zip(benign_zero_counts, malicious_kept_counts)):
                # collect ids again for clarity
                b_ids, m_ids = check_epoch(w[i])
                writer.writerow([i + 1, bz, mk, " ".join(map(str, b_ids)), " ".join(map(str, m_ids))])
        print(f"Wrote stats to {args.out_csv}")


if __name__ == "__main__":
    main()
