"""
memory.py — Driving-scene memory bank with retrieval and OOD scoring.

The memory bank stores (embedding, action, scenario_label) triples from a
set of "known" driving scenes. At inference time, given a query scene
embedding, we:

  1. Retrieve the k nearest neighbours by cosine similarity.
  2. Compute an OOD score from the similarity of the nearest neighbour:
     if the closest known scene is still far away, the query is novel.
  3. Produce a retrieval-augmented action by aggregating the actions of the
     retrieved neighbours, weighted by similarity.

This implements the "robustness to rare and novel scenarios" goal: rather
than trusting a frozen VLA model's possibly-overconfident output on a scene
unlike anything it was trained on, the system can (a) flag the scene as OOD
and (b) fall back to the verified action of the most similar known scene.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Discrete driving action space (a compact, interpretable set).
# In a full system these map to trajectory primitives.
ACTIONS = [
    "keep_lane",
    "slow_down",
    "stop",
    "turn_left",
    "turn_right",
    "change_lane_left",
    "change_lane_right",
    "yield",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)


class MemoryBank:
    """A bank of known driving-scene embeddings with their verified actions."""

    def __init__(self, embed_dim: int = 128, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.embeddings = torch.empty(0, embed_dim, device=self.device)
        self.actions = torch.empty(0, dtype=torch.long, device=self.device)
        self.scenarios: list[str] = []

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def add(self, embeddings: torch.Tensor, actions: torch.Tensor,
            scenarios: list[str]) -> None:
        """Add a batch of known scenes. embeddings [N, D], actions [N]."""
        embeddings = embeddings.to(self.device)
        actions = actions.to(self.device)
        self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self.actions = torch.cat([self.actions, actions], dim=0)
        self.scenarios.extend(scenarios)

    def retrieve(self, query: torch.Tensor, k: int = 5):
        """Return (similarities [B, k], indices [B, k]) for each query row."""
        if len(self) == 0:
            raise RuntimeError("Memory bank is empty.")
        query = query.to(self.device)
        # cosine similarity == dot product on normalised vectors
        sims = query @ self.embeddings.T              # [B, N]
        k = min(k, len(self))
        topk = torch.topk(sims, k=k, dim=-1)
        return topk.values, topk.indices

    def ood_score(self, query: torch.Tensor) -> torch.Tensor:
        """OOD score in [0, 1]; higher = more out-of-distribution.

        Defined as 1 - (similarity to nearest known scene). A query that
        sits right on top of a known scene scores ~0; a query far from
        everything scores ~1.
        """
        sims, _ = self.retrieve(query, k=1)
        nearest = sims[:, 0].clamp(-1.0, 1.0)
        return (1.0 - nearest) / 2.0                  # map [-1,1] sim -> [1,0] -> [0,1]

    def retrieved_action(self, query: torch.Tensor, k: int = 5,
                         temperature: float = 0.1):
        """Similarity-weighted vote over retrieved neighbours' actions.

        Returns (action_probs [B, N_ACTIONS], top_action_idx [B]).
        """
        sims, idxs = self.retrieve(query, k=k)        # [B, k]
        neigh_actions = self.actions[idxs]            # [B, k]
        weights = F.softmax(sims / temperature, dim=-1)  # [B, k]
        probs = torch.zeros(query.shape[0], N_ACTIONS, device=self.device)
        for a in range(N_ACTIONS):
            mask = (neigh_actions == a).float()       # [B, k]
            probs[:, a] = (weights * mask).sum(dim=-1)
        return probs, probs.argmax(dim=-1)


# Index of the conservative fallback action used when OOD and retrieval confidence
# is too low to trust any specific retrieved action.
SAFE_FALLBACK_ACTION = ACTION_TO_IDX["stop"]  # stop is safest in genuine uncertainty


def _retrieval_consensus(memory: MemoryBank, query_emb: torch.Tensor,
                         k: int) -> torch.Tensor:
    """Fraction of top-k retrieved neighbours that agree on the plurality action.

    High consensus (near 1.0): the k neighbours mostly agree — retrieval is
    trustworthy because there is a clear action signal.
    Low consensus (near 1/k): the neighbours are split across actions —
    retrieval is untrustworthy because the query sits between distinct regions
    of the action space (the tell-tale sign of a truly OOD scene).
    """
    sims, idxs = memory.retrieve(query_emb, k=k)         # [B, k]
    neigh_actions = memory.actions[idxs]                  # [B, k]
    B = query_emb.shape[0]
    consensus = torch.zeros(B, device=memory.device)
    for b in range(B):
        acts = neigh_actions[b]
        # fraction voting for the plurality action
        counts = torch.bincount(acts, minlength=N_ACTIONS)
        consensus[b] = counts.max().float() / k
    return consensus                                        # [B] in [1/k, 1.0]


def safe_action_selection(vla_action_probs: torch.Tensor,
                          memory: MemoryBank,
                          query_emb: torch.Tensor,
                          ood_threshold: float = 0.5,
                          retrieval_confidence_threshold: float = 0.85,
                          k: int = 5):
    """Three-way safe action selection with consensus gating.

    1. IN-DISTRIBUTION (OOD score < ood_threshold):
       Trust the VLA model directly.

    2. OOD + HIGH CONSENSUS (retrieved neighbours mostly agree on one action):
       The scene is novel but the memory bank has a clear action signal in
       this region of embedding space. Use the similarity-weighted retrieval vote.

    3. OOD + LOW CONSENSUS (retrieved neighbours disagree):
       The query sits between distinct action regions — a sign of genuine
       novelty where no retrieved action should be trusted. Fall back to the
       conservative safe action (slow_down).

    The consensus gate is the key insight: raw retrieval similarity is a poor
    proxy for action trustworthiness because visually similar scenes can require
    very different actions (red traffic light vs red warning sign on a wrong-way
    vehicle). Requiring agreement among the top-k neighbours filters this out.
    """
    ood = memory.ood_score(query_emb)                      # [B]
    consensus = _retrieval_consensus(memory, query_emb, k) # [B] in [1/k, 1.0]

    retr_probs, retr_action = memory.retrieved_action(query_emb, k=k)
    vla_action = vla_action_probs.argmax(dim=-1)

    is_ood = ood > ood_threshold                           # [B] bool

    # For OOD scenes, safe fallback is the DEFAULT.
    # Retrieval is only used when consensus is very high (neighbours strongly
    # agree) AND the OOD score is relatively low (scene is borderline, not
    # wildly novel). When a scene is both far from everything in memory AND
    # neighbours disagree, decelerating is always safer than a wrong manoeuvre.
    borderline_ood = (ood > ood_threshold) & (ood < ood_threshold * 3.0)
    high_consensus = consensus >= retrieval_confidence_threshold   # [B] bool
    use_retrieval_gate = borderline_ood & high_consensus

    fallback = torch.full_like(vla_action, SAFE_FALLBACK_ACTION)

    # Default: VLA for in-distribution, fallback for OOD
    final = torch.where(is_ood, fallback, vla_action)
    # Override: borderline OOD with strong consensus -> use retrieval
    final = torch.where(use_retrieval_gate, retr_action, final)

    path = torch.zeros_like(vla_action)                # 0=vla
    path = torch.where(is_ood & ~use_retrieval_gate,
                       torch.full_like(path, 2), path) # 2=fallback
    path = torch.where(use_retrieval_gate,
                       torch.ones_like(path), path)    # 1=retrieval

    return {
        "final_action": final,
        "ood_score": ood,
        "retrieval_consensus": consensus,
        "used_retrieval": use_retrieval_gate,
        "used_fallback": is_ood & ~use_retrieval_gate,
        "vla_action": vla_action,
        "retrieved_action": retr_action,
        "path": path,
    }


def _smoke_test() -> None:
    torch.manual_seed(0)
    D = 128
    mem = MemoryBank(embed_dim=D)
    # add 50 known scenes
    emb = F.normalize(torch.randn(50, D), dim=-1)
    acts = torch.randint(0, N_ACTIONS, (50,))
    mem.add(emb, acts, [f"scene_{i}" for i in range(50)])
    print(f"Memory bank size: {len(mem)}")

    # query: one in-distribution (copy of a known scene), one random/novel
    q_known = emb[3:4] + 0.01 * torch.randn(1, D)
    q_known = F.normalize(q_known, dim=-1)
    q_novel = F.normalize(torch.randn(1, D), dim=-1)
    queries = torch.cat([q_known, q_novel], dim=0)

    ood = mem.ood_score(queries)
    print(f"OOD scores: known={ood[0].item():.3f}  novel={ood[1].item():.3f}  "
          f"(known should be << novel)")

    fake_vla = F.softmax(torch.randn(2, N_ACTIONS), dim=-1)
    out = safe_action_selection(fake_vla, mem, queries, ood_threshold=0.4)
    print(f"Used retrieval: {out['used_retrieval'].tolist()}  "
          f"(expect [False, True])")


if __name__ == "__main__":
    _smoke_test()