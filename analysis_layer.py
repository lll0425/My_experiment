"""
Layer-wise FINCH score analysis
-------------------------------
讀取 `layer_div_score_*.csv`（由訓練程式輸出的一層一檔），
計算並可視化每層在不同 epoch 的惡意/良性分數差距。

使用方式：
  1) 修改 DATA_PATH 為你的實驗輸出資料夾（包含 layer_div_score_*.csv）。
  2) python analysis_layer.py
輸出：
  - plots/layer_finch_weights.png, layer_finch_bias.png
  - plots/layer_finch_heatmap.png
  - 終端列出各層惡/良分數比的首末變化。
"""

import argparse
import csv
import glob
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

# =========================
# Matplotlib 全域設定
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 全域設定
# =========================
NUM_CLIENTS = 10
BENIGN_CLIENTS = list(range(7))
EVIL_CLIENTS = list(range(7, 10))

# 改成你的實驗輸出路徑（含 layer_div_score_*.csv）
# DATA_PATH: folder that contains layer_div_score_*.csv
DATA_PATH = r"./data/label_skew/base_backdoor/0.3\fl_fashionmnist\100.0\Ours\fedfish\beta100.0_seed2"

def _infer_beta(data_path: str) -> tuple[str, str]:
    """
    儘量從 cfg.yaml 讀取 beta；若沒有就從路徑推斷。
    回傳 (beta_value, beta_tag)；若未知則給 'unknown'。
    """
    beta_val = None
    cfg_path = os.path.join(data_path, "cfg.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"beta:\s*([0-9.]+)", line)
                if m:
                    beta_val = m.group(1)
                    break
    if beta_val is None:
        parts = os.path.normpath(data_path).split(os.sep)
        for p in reversed(parts):
            if p.startswith("beta"):
                beta_val = p.replace("beta", "")
                break
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", p):
                beta_val = p
                break
    if not beta_val:
        beta_val = "unknown"
    beta_tag = f"beta{beta_val.replace('.', '_')}"
    return beta_val, beta_tag


BETA_VAL, BETA_TAG = _infer_beta(DATA_PATH)

PLOT_DIR = os.path.join(DATA_PATH, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# =====================================================================
# 0. aggregation_weight / attack_success plotting helpers
# =====================================================================
def _default_if_exists(path: str | None) -> str | None:
    if path and os.path.exists(path):
        return path
    return None

def load_agg_weights(csv_path: str) -> np.ndarray:
    rows = []
    maxlen = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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


def plot_malicious_detection_rate(agg_path: str):
    weights = load_agg_weights(agg_path)
    epochs = np.arange(1, len(weights) + 1)
    mal_weights = weights[:, 7:]
    mal_detected = (mal_weights == 0).sum(axis=1)
    mal_rate = mal_detected / mal_weights.shape[1]
    out_path = os.path.join(os.path.dirname(agg_path), "malicious_detection_rate.png")

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, mal_rate, color="red", linewidth=2, label="Malicious detection rate")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Epoch")
    plt.ylabel("Detection rate (0~1)")
    plt.title("Malicious detection rate per epoch (clients 7-9)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_malicious_removed_count(agg_path: str):
    weights = load_agg_weights(agg_path)
    epochs = np.arange(1, len(weights) + 1)
    mal_weights = weights[:, 7:]
    mal_detected = (mal_weights == 0).sum(axis=1)
    out_path = os.path.join(os.path.dirname(agg_path), "malicious_removed_count.png")

    colors = ["#c00000" if x < 3 else "#4f81bd" for x in mal_detected]
    plt.figure(figsize=(10, 4))
    plt.bar(epochs, mal_detected, color=colors)
    plt.yticks([0, 1, 2, 3])
    plt.xlabel("Epoch")
    plt.ylabel("# Malicious removed (out of 3)")
    plt.title("Malicious removal count per epoch")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_removal_errors(agg_path: str):
    weights = load_agg_weights(agg_path)
    epochs = np.arange(1, len(weights) + 1)
    mal_weights = weights[:, 7:]
    ben_weights = weights[:, :7]
    benign_zero_counts = (ben_weights == 0).sum(axis=1)
    malicious_kept_counts = (mal_weights > 0).sum(axis=1)
    out_path = os.path.join(os.path.dirname(agg_path), "removal_errors.png")

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, benign_zero_counts, label="False positives (benign removed)", color="steelblue")
    plt.plot(epochs, malicious_kept_counts, label="False negatives (malicious kept)", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.title("Removal errors per epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_removal_matrix(agg_path: str):
    weights = load_agg_weights(agg_path)
    kept = (weights > 0).astype(int)
    kept_map = kept.T

    error = np.zeros_like(kept_map)
    benign_ids = np.arange(0, 7)
    malicious_ids = np.arange(7, 10)
    error[benign_ids, :] = (kept_map[benign_ids, :] == 0).astype(int)
    error[malicious_ids, :] = (kept_map[malicious_ids, :] == 1).astype(int) * 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    cmap_kept = ListedColormap(["#dddddd", "#4f81bd"])
    axes[0].imshow(kept_map, aspect="auto", cmap=cmap_kept, interpolation="nearest")
    axes[0].set_title("Kept(blue) vs Removed(gray)")
    axes[0].set_ylabel("Client ID")
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels([str(i) for i in range(10)])

    cmap_err = ListedColormap(["#ffffff", "#f4b183", "#c00000"])
    axes[1].imshow(error, aspect="auto", cmap=cmap_err, interpolation="nearest")
    axes[1].set_title("Error map: FP(orange)=benign removed, FN(red)=malicious kept")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Client ID")
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([str(i) for i in range(10)])

    if kept_map.shape[1] <= 50:
        axes[1].set_xticks(range(kept_map.shape[1]))
        axes[1].set_xticklabels([str(i + 1) for i in range(kept_map.shape[1])], rotation=90, fontsize=6)
    else:
        step = max(1, kept_map.shape[1] // 10)
        axes[1].set_xticks(range(0, kept_map.shape[1], step))
        axes[1].set_xticklabels([str(i + 1) for i in range(0, kept_map.shape[1], step)])

    out_path = os.path.join(os.path.dirname(agg_path), "removal_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_attack_success(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    if len(rows) < 2:
        raise ValueError("CSV should have header row and one data row.")
    values = [float(x) for x in rows[1]]
    epochs = list(range(len(values)))
    out_path = os.path.join(os.path.dirname(csv_path), "attack_success_rate.png")
    non_iid, _ = _infer_beta(os.path.dirname(csv_path))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, values, color="#c00000", linewidth=2)
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Attack success rate (%)")
    plt.title(f"Attack Sucess Rate (non-iid={non_iid})")
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


def run_all(data_path: str = DATA_PATH,
            agg_path: str | None = None,
            attack_csv: str | None = None,
            skip_layer: bool = False):
    # auto-pick default files if not provided
    if agg_path is None:
        agg_path = _default_if_exists(os.path.join(data_path, "aggregation_weight.csv"))
    if attack_csv is None:
        attack_csv = _default_if_exists(os.path.join(data_path, "attack_success_rate_mean_acc.csv"))

    if not skip_layer:
        global BETA_VAL, BETA_TAG, PLOT_DIR
        BETA_VAL, BETA_TAG = _infer_beta(data_path)
        PLOT_DIR = os.path.join(data_path, "plots")
        os.makedirs(PLOT_DIR, exist_ok=True)

        layer_evolution = load_layer_scores(data_path, prefix="layer_div_score")
        if not layer_evolution:
            raise SystemExit("No layer_div_score files found.")

        print(f"beta={BETA_VAL}, data_path={data_path}")

        base = os.path.join(PLOT_DIR, "layer_finch")

        print("[Param x Fisher]")
        print_summary(layer_evolution)
        export_layer_ratio_analysis(
            layer_evolution,
            analysis_name="Param x Fisher",
            file_tag="weighted",
            out_dir=PLOT_DIR,
        )
        plot_evolution_curves(layer_evolution, base)
        plot_heatmap(
            layer_evolution,
            base,
            analysis_name="Param x Fisher",
            file_tag="weighted",
        )

        extra_heatmaps = [
            ("layer_param_div_score", "Parameter Only", "param"),
            ("layer_fisher_div_score", "Fisher Only", "fisher"),
        ]
        for prefix, analysis_name, tag in extra_heatmaps:
            extra_layer_evolution = load_layer_scores(data_path, prefix=prefix)
            if not extra_layer_evolution:
                print(f"Skip {prefix}: no files found.")
                continue
            print(f"[{analysis_name}]")
            print_summary(extra_layer_evolution)
            export_layer_ratio_analysis(
                extra_layer_evolution,
                analysis_name=analysis_name,
                file_tag=tag,
                out_dir=PLOT_DIR,
            )
            plot_heatmap(
                extra_layer_evolution,
                base,
                analysis_name=analysis_name,
                file_tag=tag,
            )

        print("plots saved to:", PLOT_DIR)

    if agg_path:
        plot_malicious_detection_rate(agg_path)
        plot_malicious_removed_count(agg_path)
        plot_removal_errors(agg_path)
        plot_removal_matrix(agg_path)

    if attack_csv:
        plot_attack_success(attack_csv)

# =====================================================================
# 1. 讀取每層、每輪、每個 client 的分數
# =====================================================================
def load_layer_scores(data_path: str, prefix: str = "layer_div_score") -> Dict[str, Dict[int, dict]]:
    layer_files = sorted(glob.glob(os.path.join(data_path, f"{prefix}_*.csv")))
    if not layer_files:
        print(f"No {prefix}_*.csv files found in: {data_path}")
        return {}

    layer_evolution = {}
    for fp in layer_files:
        layer = os.path.basename(fp).replace(f"{prefix}_", "").replace(".csv", "")
        df = pd.read_csv(fp)
        # 期待格式：epoch, client_0, client_1, ...
        epochs = df.iloc[:, 0].to_numpy()
        scores = df.iloc[:, 1:].to_numpy()  # shape: (n_epoch, n_client)
        layer_evolution[layer] = {
            int(e): {
                "all_scores": scores[i],
                "benign_scores": scores[i][BENIGN_CLIENTS],
                "evil_scores": scores[i][EVIL_CLIENTS],
            }
            for i, e in enumerate(epochs)
        }
    return layer_evolution


# =====================================================================
# 2. 曲線圖：Weight / Bias 分開
# =====================================================================
def plot_evolution_curves(layer_evolution: dict, save_prefix: str | None = None):
    layer_names = list(layer_evolution.keys())
    weight_layers = [l for l in layer_names if "weight" in l]
    bias_layers = [l for l in layer_names if "bias" in l]

    if weight_layers:
        _plot_layer_group(
            layer_evolution,
            weight_layers,
            "Weight Layers",
            None if not save_prefix else save_prefix + f"_{BETA_TAG}_weights.png",
        )
    if bias_layers:
        _plot_layer_group(
            layer_evolution,
            bias_layers,
            "Bias Layers",
            None if not save_prefix else save_prefix + f"_{BETA_TAG}_bias.png",
        )


def _plot_layer_group(layer_evolution, layer_names, title, save_path=None):
    n_cols = 3
    n_rows = (len(layer_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, layer in enumerate(layer_names):
        ax = axes[idx // n_cols, idx % n_cols]
        data = layer_evolution[layer]
        epochs = sorted(data.keys())

        for cid in BENIGN_CLIENTS:
            ax.plot(
                epochs,
                [data[e]["all_scores"][cid] for e in epochs],
                color="blue",
                alpha=0.35,
            )
        for cid in EVIL_CLIENTS:
            ax.plot(
                epochs,
                [data[e]["all_scores"][cid] for e in epochs],
                color="red",
                alpha=0.55,
            )

        ax.plot(
            epochs,
            [data[e]["benign_scores"].mean() for e in epochs],
            color="blue",
            linestyle="--",
            linewidth=2,
            label="Benign Mean",
        )
        ax.plot(
            epochs,
            [data[e]["evil_scores"].mean() for e in epochs],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Malicious Mean",
        )

        ax.set_title(layer)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FINCH Score")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend()

    # 隱藏多餘子圖
    for idx in range(len(layer_names), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(f"{title} FINCH Score Evolution (beta={BETA_VAL})", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =====================================================================
# 3. 熱力圖：惡/良分數比
# =====================================================================
def plot_heatmap(
    layer_evolution: dict,
    save_prefix: str | None = None,
    analysis_name: str = "Param x Fisher",
    file_tag: str = "weighted",
):
    layer_names = list(layer_evolution.keys())
    weight_layers = [l for l in layer_names if "weight" in l]
    bias_layers = [l for l in layer_names if "bias" in l]

    if weight_layers:
        _plot_heatmap_group(
            layer_evolution,
            weight_layers,
            f"Weight Layers - Malicious / Benign {analysis_name} Ratio",
            None if not save_prefix else save_prefix + f"_{BETA_TAG}_{file_tag}_heatmap_weights.png",
        )

    if bias_layers:
        _plot_heatmap_group(
            layer_evolution,
            bias_layers,
            f"Bias Layers - Malicious / Benign {analysis_name} Ratio",
            None if not save_prefix else save_prefix + f"_{BETA_TAG}_{file_tag}_heatmap_bias.png",
        )


def _plot_heatmap_group(layer_evolution, layer_names, title, save_path=None):
    """畫出指定 layer 群組的惡/良分數比熱力圖。"""
    max_epochs = max(len(layer_evolution[l]) for l in layer_names)
    matrix = np.full((len(layer_names), max_epochs), np.nan)

    for i, layer in enumerate(layer_names):
        for e, data in layer_evolution[layer].items():
            matrix[i, int(e)] = data["evil_scores"].mean() / (data["benign_scores"].mean() + 1e-8)

    fig, ax = plt.subplots(figsize=(15, max(6, 0.5 * len(layer_names))))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(max_epochs))
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")
    ax.set_title(title, fontsize=16)

    cbar = plt.colorbar(im)
    cbar.set_label("Separation Ratio", rotation=270, labelpad=20)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =====================================================================
# 4. 文字摘要
# =====================================================================
def print_summary(layer_evolution: dict):
    print("\nFINCH Separation Summary\n" + "=" * 60)
    for layer, data in layer_evolution.items():
        epochs = sorted(data.keys())
        f, l = epochs[0], epochs[-1]
        fr = data[f]["evil_scores"].mean() / (data[f]["benign_scores"].mean() + 1e-8)
        lr = data[l]["evil_scores"].mean() / (data[l]["benign_scores"].mean() + 1e-8)
        print(f"{layer:30s}  {fr:.2f} -> {lr:.2f}  d={lr - fr:.2f}")


def _ratio_series_by_layer(layer_evolution: dict) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for layer, data in layer_evolution.items():
        epochs = np.array(sorted(data.keys()), dtype=float)
        ratios = np.array(
            [
                data[int(e)]["evil_scores"].mean() / (data[int(e)]["benign_scores"].mean() + 1e-8)
                for e in epochs
            ],
            dtype=float,
        )
        out[layer] = (epochs, ratios)
    return out


def export_layer_ratio_analysis(
    layer_evolution: dict,
    analysis_name: str,
    file_tag: str,
    out_dir: str,
):
    ratio_map = _ratio_series_by_layer(layer_evolution)
    if not ratio_map:
        return

    n_epochs = len(next(iter(ratio_map.values()))[0])
    idx_groups = np.array_split(np.arange(n_epochs), 3)
    stage_names = ["early", "middle", "late"]

    rows = []
    for layer, (epochs, ratios) in ratio_map.items():
        stage_medians = []
        stage_ranges = {}
        for sname, idx in zip(stage_names, idx_groups):
            vals = ratios[idx]
            stage_medians.append(float(np.median(vals)))
            stage_ranges[sname] = f"{int(epochs[idx[0]])}-{int(epochs[idx[-1]])}"

        slope = float(np.polyfit(epochs, ratios, 1)[0]) if len(epochs) > 1 else 0.0
        rows.append(
            {
                "layer": layer,
                "overall_median": float(np.median(ratios)),
                "early_median": stage_medians[0],
                "middle_median": stage_medians[1],
                "late_median": stage_medians[2],
                "late_minus_early": stage_medians[2] - stage_medians[0],
                "slope_per_epoch": slope,
                "early_epoch_range": stage_ranges["early"],
                "middle_epoch_range": stage_ranges["middle"],
                "late_epoch_range": stage_ranges["late"],
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["overall_median", "late_median", "slope_per_epoch"],
        ascending=[False, False, False],
    )
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    print(f"\nLayer Ratio Trend Table [{analysis_name}]")
    print("-" * 90)
    cols = [
        "rank",
        "layer",
        "overall_median",
        "early_median",
        "middle_median",
        "late_median",
        "late_minus_early",
        "slope_per_epoch",
    ]
    print(df[cols].to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))
    print(
        "epoch split:",
        f"early={df['early_epoch_range'].iloc[0]},",
        f"middle={df['middle_epoch_range'].iloc[0]},",
        f"late={df['late_epoch_range'].iloc[0]}",
    )

    out_path = os.path.join(out_dir, f"layer_ratio_trend_{BETA_TAG}_{file_tag}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved ratio trend table to {out_path}")


# =====================================================================
# 5. main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all plots from one script.")
    parser.add_argument("--data_path", default=DATA_PATH, help="Folder with layer_div_score_*.csv")
    parser.add_argument("--agg_path", default=None, help="Path to aggregation_weight.csv")
    parser.add_argument("--attack_csv", default=None, help="Path to attack_success_rate_mean_acc.csv")
    parser.add_argument("--skip_layer", action="store_true", help="Skip layer_div_score plots")
    args = parser.parse_args()

    run_all(
        data_path=args.data_path,
        agg_path=args.agg_path,
        attack_csv=args.attack_csv,
        skip_layer=args.skip_layer,
    )
