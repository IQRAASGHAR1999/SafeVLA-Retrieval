"""
visualize.py — Generate result figures for SafeVLA-Retrieval.

Produces:
  1. scenario_gallery.png  — example synthetic scenes (common + rare)
  2. embedding_tsne.png     — 2D t-SNE of scene embeddings coloured by action,
                              with rare scenarios overlaid to show they fall
                              outside the common clusters
  3. ood_histogram.png      — OOD-score distributions for common vs rare scenes
  4. accuracy_comparison.png— bar chart: VLA-only vs retrieval-augmented
                              accuracy on common and rare splits
  5. retrieval_example.png  — a rare query with its top-k retrieved neighbours

Usage:
    python visualize.py --checkpoint runs/exp1/checkpoint.pt --data data
    python visualize.py --gallery-only --data data    # just the scene gallery
    python visualize.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from encoder import SceneEncoder
from memory import MemoryBank, ACTIONS, N_ACTIONS, safe_action_selection


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def plot_scenario_gallery(data_dir, out_dir):
    common = torch.load(Path(data_dir) / "common.pt", weights_only=False)
    rare = torch.load(Path(data_dir) / "rare.pt", weights_only=False)

    # one example per unique scenario
    def first_of_each(d):
        seen, picks = {}, []
        for i, s in enumerate(d["scenarios"]):
            if s not in seen:
                seen[s] = i
                picks.append((s, i))
        return picks

    common_ex = first_of_each(common)
    rare_ex = first_of_each(rare)
    all_ex = [("COMMON", common, common_ex), ("RARE", rare, rare_ex)]

    n_cols = max(len(common_ex), len(rare_ex))
    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))
    fig.suptitle("Synthetic driving scenarios: common (top) vs rare (bottom)",
                 fontsize=12, weight="bold")

    for row, (label, d, exs) in enumerate(all_ex):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(exs):
                name, idx = exs[col]
                img = d["images"][idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(name.replace("_", " "), fontsize=8)
    _save(fig, out_dir / "scenario_gallery.png")


def plot_embeddings(enc, common, rare, device, out_dir):
    from sklearn.manifold import TSNE
    enc.eval()
    with torch.no_grad():
        c_emb = enc(common["images"].to(device)).cpu().numpy()
        r_emb = enc(rare["images"].to(device)).cpu().numpy()

    all_emb = np.vstack([c_emb, r_emb])
    tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca")
    proj = tsne.fit_transform(all_emb)
    c_proj = proj[:len(c_emb)]
    r_proj = proj[len(c_emb):]

    fig, ax = plt.subplots(figsize=(8, 6))
    c_acts = common["actions"].numpy()
    sc = ax.scatter(c_proj[:, 0], c_proj[:, 1], c=c_acts, cmap="tab10",
                    s=18, alpha=0.6, label="common")
    ax.scatter(r_proj[:, 0], r_proj[:, 1], c="black", marker="x", s=60,
               linewidths=2, label="rare (held out)")
    ax.set_title("Scene embedding space (t-SNE)\n"
                 "colour = action class; black x = rare scenarios",
                 fontsize=11, weight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    _save(fig, out_dir / "embedding_tsne.png")


def plot_ood_histogram(c_ood, r_ood, out_dir, threshold):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 30)
    ax.hist(c_ood, bins=bins, alpha=0.7, label="common (in-distribution)",
            color="#2196F3", density=True)
    ax.hist(r_ood, bins=bins, alpha=0.7, label="rare (out-of-distribution)",
            color="#F44336", density=True)
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"OOD threshold = {threshold}")
    ax.set_xlabel("OOD score (1 - similarity to nearest known scene)")
    ax.set_ylabel("Density")
    ax.set_title("OOD score separates rare scenarios from common ones",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    _save(fig, out_dir / "ood_histogram.png")


def plot_accuracy(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("SafeVLA-Retrieval: three-way safe action selection",
                 fontsize=12, weight="bold")

    # Left: accuracy comparison
    ax = axes[0]
    groups = ["Common\n(in-distribution)", "Rare\n(OOD)"]
    vla  = [results["common_vla_acc"],  results["rare_vla_acc"]]
    safe = [results["common_safe_acc"], results["rare_safe_acc"]]
    x = np.arange(len(groups))
    w = 0.35
    b1 = ax.bar(x - w/2, vla,  w, label="VLA only",
                color="#90A4AE", zorder=3)
    b2 = ax.bar(x + w/2, safe, w, label="SafeVLA-Retrieval (ours)",
                color="#43A047", zorder=3)
    ax.set_ylabel("Action accuracy")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.set_title("Action accuracy", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=8)

    # Right: decision path breakdown for rare scenes
    ax2 = axes[1]
    vla_frac  = 1.0 - results.get("rare_flagged_frac", 0.0)
    retr_frac = results.get("rare_retrieval_frac", 0.0)
    fall_frac = results.get("rare_fallback_frac", 0.0)
    wedges, texts, autotexts = ax2.pie(
        [vla_frac, retr_frac, fall_frac],
        labels=["VLA trusted\n(in-distribution)",
                "Retrieval used\n(OOD + high conf)",
                "Safe fallback\n(OOD + low conf)"],
        colors=["#90A4AE", "#1E88E5", "#E53935"],
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 8},
    )
    ax2.set_title("Decision path for rare scenes", fontsize=10)

    _save(fig, out_dir / "accuracy_comparison.png")


def plot_retrieval_example(enc, mem, rare, common, device, out_dir, k=5):
    enc.eval()
    with torch.no_grad():
        q_emb = enc(rare["images"][:1].to(device))
        sims, idxs = mem.retrieve(q_emb, k=k)
    sims = sims[0].cpu().numpy()
    idxs = idxs[0].cpu().numpy()

    fig, axes = plt.subplots(1, k + 1, figsize=(2.0 * (k + 1), 2.6))
    q_img = rare["images"][0].permute(1, 2, 0).numpy()
    axes[0].imshow(q_img)
    axes[0].set_title(f"QUERY (rare)\n{rare['scenarios'][0].replace('_',' ')}",
                      fontsize=8, color="#C62828")
    axes[0].axis("off")

    mem_imgs = common["images"]
    # Map memory indices back to original common indices
    ckpt_scen = mem.scenarios
    for j in range(k):
        ax = axes[j + 1]
        # find a common image matching the retrieved scenario
        scen = ckpt_scen[idxs[j]]
        match = next(i for i, s in enumerate(common["scenarios"]) if s == scen)
        ax.imshow(common["images"][match].permute(1, 2, 0).numpy())
        ax.set_title(f"#{j+1}  sim={sims[j]:.2f}\n{scen.replace('_',' ')}",
                     fontsize=7)
        ax.axis("off")
    fig.suptitle("Retrieval for a rare query: nearest known scenes and their verified actions",
                 fontsize=10, weight="bold")
    _save(fig, out_dir / "retrieval_example.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="runs/exp1/checkpoint.pt")
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="docs/figures")
    p.add_argument("--gallery-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ood-threshold", type=float, default=0.5)
    args = p.parse_args()

    if not HAS_PLOT:
        print("matplotlib/numpy/sklearn needed: pip install matplotlib scikit-learn")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_test:
        import tempfile
        from dataset import generate
        tmp = tempfile.mkdtemp()
        generate(tmp, n_per_common=15, n_per_rare=6)
        args.data = tmp
        plot_scenario_gallery(args.data, out_dir)
        print("Smoke test done (gallery only).")
        return

    # Gallery needs only the data
    plot_scenario_gallery(args.data, out_dir)
    if args.gallery_only:
        return

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    enc = SceneEncoder(embed_dim=ckpt["args"]["embed_dim"]).to(device)
    enc.load_state_dict(ckpt["encoder"])

    mem = MemoryBank(embed_dim=ckpt["args"]["embed_dim"], device=str(device))
    mem.add(ckpt["mem_embeddings"], ckpt["mem_actions"], ckpt["mem_scenarios"])

    common = torch.load(Path(args.data) / "common.pt", weights_only=False)
    rare = torch.load(Path(args.data) / "rare.pt", weights_only=False)

    # embeddings + OOD
    enc.eval()
    with torch.no_grad():
        c_emb = enc(common["images"].to(device))
        r_emb = enc(rare["images"].to(device))
    c_ood = mem.ood_score(c_emb).cpu().numpy()
    r_ood = mem.ood_score(r_emb).cpu().numpy()

    # Use threshold from saved results if available (adaptive), else CLI arg
    threshold = args.ood_threshold
    res_path_t = Path(args.checkpoint).parent / "results.json"
    if res_path_t.exists():
        with open(res_path_t) as ft:
            saved = json.load(ft)
            threshold = saved.get("results", {}).get("ood_threshold_used", threshold)

    plot_embeddings(enc, common, rare, device, out_dir)
    plot_ood_histogram(c_ood, r_ood, out_dir, threshold)
    plot_retrieval_example(enc, mem, rare, common, device, out_dir, k=args.k)

    # results
    res_path = Path(args.checkpoint).parent / "results.json"
    if res_path.exists():
        with open(res_path) as f:
            results = json.load(f)["results"]
        plot_accuracy(results, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()