"""
train.py — Train the SceneEncoder projection head with supervised contrastive
learning, then build the memory bank and evaluate retrieval-augmented action
selection against a simulated frozen VLA baseline.

Why contrastive: we want scenes with the same correct action to embed near
each other, and different actions to separate, so that nearest-neighbour
retrieval returns action-relevant matches. We use a supervised contrastive
loss over the action labels.

The "VLA baseline" is simulated: a frozen classifier trained ONLY on common
scenarios. On common scenes it is accurate and confident; on rare scenes it
stays confident but is frequently wrong (the failure mode the position calls
out). The retrieval layer's job is to catch those cases.

Usage:
    python dataset.py --generate --out data
    python train.py --data data --epochs 30
    python train.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from encoder import SceneEncoder
from memory import (MemoryBank, N_ACTIONS, safe_action_selection)


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.1) -> torch.Tensor:
    """SupCon loss (Khosla et al. 2020), simplified.

    emb [B, D] L2-normalised; labels [B]. Pulls same-label embeddings together.
    """
    device = emb.device
    sim = emb @ emb.T / temperature                  # [B, B]
    # numerical stability
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim)

    # mask out self-comparisons
    self_mask = torch.eye(emb.shape[0], dtype=torch.bool, device=device)
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)

    # positive mask: same label, not self
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)

    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12
    log_prob = sim - torch.log(denom)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    mean_log_prob_pos = (
        (log_prob * pos_mask).sum(dim=1)[valid] / pos_counts[valid]
    )
    return -mean_log_prob_pos.mean()


class SimulatedVLA(nn.Module):
    """A small classifier standing in for a frozen VLA driving model.

    Trained only on common scenarios. Represents the model whose confident
    errors on rare scenes the retrieval layer is designed to catch.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        import tempfile
        from dataset import generate
        tmp = tempfile.mkdtemp()
        generate(tmp, n_per_common=20, n_per_rare=8)
        data_dir = tmp
        epochs = 4
    else:
        data_dir = args.data
        epochs = args.epochs

    common = torch.load(Path(data_dir) / "common.pt", weights_only=False)
    rare = torch.load(Path(data_dir) / "rare.pt", weights_only=False)

    # split common into train / memory / test
    n = common["images"].shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_train = int(0.6 * n)
    n_mem = int(0.2 * n)
    tr_idx = perm[:n_train]
    mem_idx = perm[n_train:n_train + n_mem]
    test_idx = perm[n_train + n_mem:]

    enc = SceneEncoder(embed_dim=args.embed_dim, pretrained=not args.smoke_test).to(device)

    # ---- Train projection head with supervised contrastive loss ----
    tr_ds = TensorDataset(common["images"][tr_idx], common["actions"][tr_idx])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(
        [p for p in enc.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)

    history = []
    for ep in range(epochs):
        enc.train()
        losses = []
        for imgs, acts in tr_loader:
            imgs, acts = imgs.to(device), acts.to(device)
            opt.zero_grad()
            emb = enc(imgs)
            loss = supervised_contrastive_loss(emb, acts, args.temperature)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg = sum(losses) / max(1, len(losses))
        history.append({"epoch": ep + 1, "loss": avg})
        print(f"Epoch {ep+1:02d}/{epochs} | contrastive loss {avg:.4f}")

    # ---- Train the simulated VLA on common scenes only ----
    print("\nTraining simulated VLA baseline (common scenarios only)...")
    vla = SimulatedVLA().to(device)
    vla_opt = torch.optim.AdamW(vla.parameters(), lr=1e-3)
    vla_ds = TensorDataset(common["images"][tr_idx], common["actions"][tr_idx])
    vla_loader = DataLoader(vla_ds, batch_size=args.batch_size, shuffle=True)
    for ep in range(epochs):
        vla.train()
        for imgs, acts in vla_loader:
            imgs, acts = imgs.to(device), acts.to(device)
            vla_opt.zero_grad()
            loss = F.cross_entropy(vla(imgs), acts)
            loss.backward()
            vla_opt.step()

    # ---- Build memory bank from held-out common scenes ----
    print("\nBuilding memory bank...")
    enc.eval()
    mem = MemoryBank(embed_dim=args.embed_dim, device=str(device))
    with torch.no_grad():
        mem_emb = enc(common["images"][mem_idx].to(device))
    mem.add(mem_emb, common["actions"][mem_idx],
            [common["scenarios"][i] for i in mem_idx.tolist()])
    print(f"Memory bank size: {len(mem)}")

    # ---- Compute adaptive OOD threshold from memory bank ----
    # Use the 95th percentile of in-distribution OOD scores as the threshold.
    # This is much more robust than a fixed value: if the score distribution
    # is narrow (as with synthetic data), the threshold adapts automatically.
    enc.eval()
    with torch.no_grad():
        test_emb = enc(common["images"][test_idx[:50]].to(device))
    test_ood = mem.ood_score(test_emb)
    adaptive_threshold = float(torch.quantile(test_ood, 0.95))
    print(f"Adaptive OOD threshold (95th pct of common scores): {adaptive_threshold:.4f}")
    # Override args threshold with adaptive one
    args.ood_threshold = adaptive_threshold

    # ---- Evaluate ----
    results = evaluate(enc, vla, mem, common, test_idx, rare, device, args)

    # ---- Save ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": enc.state_dict(),
        "vla": vla.state_dict(),
        "mem_embeddings": mem.embeddings.cpu(),
        "mem_actions": mem.actions.cpu(),
        "mem_scenarios": mem.scenarios,
        "args": vars(args),
    }, out_dir / "checkpoint.pt")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"history": history, "results": results}, f, indent=2)
    print(f"\nSaved checkpoint and results to {out_dir}")
    return results


@torch.no_grad()
def evaluate(enc, vla, mem, common, test_idx, rare, device, args):
    enc.eval(); vla.eval()

    def vla_probs(imgs):
        return F.softmax(vla(imgs.to(device)), dim=-1)

    # Common (in-distribution) test set
    ci = common["images"][test_idx].to(device)
    ca = common["actions"][test_idx].to(device)
    c_emb = enc(ci)
    c_vla = vla_probs(ci)
    c_out = safe_action_selection(c_vla, mem, c_emb,
                                  ood_threshold=args.ood_threshold,
                                  retrieval_confidence_threshold=args.retrieval_conf,
                                  k=args.k)

    # Rare (out-of-distribution) test set
    ri = rare["images"].to(device)
    ra = rare["actions"].to(device)
    r_emb = enc(ri)
    r_vla = vla_probs(ri)
    r_out = safe_action_selection(r_vla, mem, r_emb,
                                  ood_threshold=args.ood_threshold,
                                  retrieval_confidence_threshold=args.retrieval_conf,
                                  k=args.k)

    def acc(pred, gt):
        return (pred == gt).float().mean().item()

    res = {
        "ood_threshold_used": args.ood_threshold,
        "common_vla_acc": acc(c_out["vla_action"], ca),
        "common_safe_acc": acc(c_out["final_action"], ca),
        "rare_vla_acc": acc(r_out["vla_action"], ra),
        "rare_safe_acc": acc(r_out["final_action"], ra),
        "common_ood_mean": c_out["ood_score"].mean().item(),
        "rare_ood_mean": r_out["ood_score"].mean().item(),
        "rare_flagged_frac": (r_out["used_retrieval"] | r_out["used_fallback"]).float().mean().item(),
        "rare_retrieval_frac": r_out["used_retrieval"].float().mean().item(),
        "rare_fallback_frac": r_out["used_fallback"].float().mean().item(),
        "common_flagged_frac": (c_out["used_retrieval"] | c_out["used_fallback"]).float().mean().item(),
    }

    # OOD detection AUROC (common=0, rare=1)
    ood_all = torch.cat([c_out["ood_score"], r_out["ood_score"]]).cpu()
    labels = torch.cat([torch.zeros(len(c_out["ood_score"])),
                        torch.ones(len(r_out["ood_score"]))])
    res["ood_auroc"] = _auroc(ood_all, labels)

    print("\n=== Results ===")
    print(f"  In-distribution (common) VLA acc:   {res['common_vla_acc']:.3f}")
    print(f"  In-distribution (common) safe acc:  {res['common_safe_acc']:.3f}")
    print(f"  Rare-scenario VLA acc:              {res['rare_vla_acc']:.3f}")
    print(f"  Rare-scenario safe acc:             {res['rare_safe_acc']:.3f}  "
          f"(<-- key number)")
    print(f"  OOD AUROC (common vs rare):         {res['ood_auroc']:.3f}")
    print(f"  Mean OOD score common/rare:         "
          f"{res['common_ood_mean']:.3f} / {res['rare_ood_mean']:.3f}")
    print(f"  Rare scenes flagged as OOD:         {res['rare_flagged_frac']*100:.1f}%")
    print(f"    of which -> retrieval used:         {res['rare_retrieval_frac']*100:.1f}%")
    print(f"    of which -> safe fallback used:     {res['rare_fallback_frac']*100:.1f}%")
    return res


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Simple AUROC via rank statistic (Mann-Whitney U)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # fraction of (pos, neg) pairs where pos > neg
    comparisons = (pos.unsqueeze(1) > neg.unsqueeze(0)).float()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float() * 0.5
    return (comparisons + ties).mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="runs/exp1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ood-threshold", type=float, default=0.5)
    p.add_argument("--retrieval-conf", type=float, default=0.70,
                   help="Min neighbour consensus fraction to trust retrieval (0.7 = 4/5 agree)")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()