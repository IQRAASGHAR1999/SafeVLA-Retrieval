"""
dataset.py — Synthetic driving-scene dataset with controllable rare scenarios.

Real VLA datasets (nuScenes, Waymo, Bench2Drive) are large and require
download/registration. To make this repository runnable end-to-end on any
machine, we generate a synthetic dataset of simple top-down driving scenes
rendered as small RGB images, each labelled with the correct discrete action
and a scenario tag.

The dataset deliberately separates COMMON scenarios (used to populate the
memory bank and train the projection head) from RARE scenarios (held out, to
test OOD detection and retrieval-augmented action selection). This mirrors
the real research problem: a model trained on common driving must behave
safely when it meets a rare event it never saw.

Common scenarios:  empty_road, car_ahead, red_light, pedestrian_crossing,
                    left_turn, right_turn
Rare scenarios:     wrong_way_vehicle, debris_on_road, animal_crossing,
                    flooded_road, fallen_cyclist

Each scene is a 64x64 RGB top-down render. This is intentionally schematic;
the point is the retrieval/OOD methodology, not photorealism. The encoder
and memory mechanism transfer unchanged to real image features.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from memory import ACTION_TO_IDX

IMG = 64

COMMON_SCENARIOS = {
    "empty_road":           "keep_lane",
    "car_ahead":            "slow_down",
    "red_light":            "stop",
    "pedestrian_crossing":  "stop",
    "left_turn":            "turn_left",
    "right_turn":           "turn_right",
}
RARE_SCENARIOS = {
    "wrong_way_vehicle":    "stop",       # stop and wait rather than lane change into unknown
    "debris_on_road":       "stop",       # full stop safer than partial lane change
    "animal_crossing":      "stop",
    "flooded_road":         "stop",       # stop before entering flooded section
    "fallen_cyclist":       "stop",       # stop and call for help
}


def _road(canvas: torch.Tensor) -> None:
    """Draw a grey vertical road with lane markings on a green background."""
    canvas[1, :, :] = 0.45                     # green base (grass)
    canvas[:, :, 24:40] = 0.25                 # grey road
    canvas[0, :, 24:40] = 0.25
    canvas[1, :, 24:40] = 0.25
    canvas[2, :, 24:40] = 0.25
    # dashed centre line (yellow)
    for y in range(0, IMG, 8):
        canvas[0, y:y+4, 31:33] = 0.9
        canvas[1, y:y+4, 31:33] = 0.8
        canvas[2, y:y+4, 31:33] = 0.1


def _box(canvas, cy, cx, h, w, rgb):
    y0, y1 = max(0, cy - h // 2), min(IMG, cy + h // 2)
    x0, x1 = max(0, cx - w // 2), min(IMG, cx + w // 2)
    for c in range(3):
        canvas[c, y0:y1, x0:x1] = rgb[c]


def render_scene(scenario: str, rng: random.Random) -> torch.Tensor:
    """Render a 3x64x64 schematic top-down scene for a scenario."""
    canvas = torch.zeros(3, IMG, IMG)
    _road(canvas)
    j = lambda: rng.randint(-2, 2)             # small jitter

    if scenario == "empty_road":
        pass
    elif scenario == "car_ahead":
        _box(canvas, 24 + j(), 32 + j(), 10, 8, (0.2, 0.3, 0.85))
    elif scenario == "red_light":
        _box(canvas, 8 + j(), 32 + j(), 6, 6, (0.9, 0.1, 0.1))
    elif scenario == "pedestrian_crossing":
        for x in range(24, 40, 4):
            canvas[:, 30:34, x:x+2] = 0.95
        _box(canvas, 32 + j(), 28 + j(), 4, 3, (0.95, 0.8, 0.6))
    elif scenario == "left_turn":
        canvas[:, 30:34, 8:32] = 0.25
    elif scenario == "right_turn":
        canvas[:, 30:34, 32:56] = 0.25
    # rare — deliberately distinctive: large, multi-element, unusual colors/layout
    elif scenario == "wrong_way_vehicle":
        # Oncoming vehicle PLUS warning markers on both sides of road
        _box(canvas, 18 + j(), 32 + j(), 12, 10, (0.95, 0.05, 0.05))   # large bright red
        _box(canvas, 10 + j(), 20 + j(), 5, 5, (1.0, 0.8, 0.0))        # left warning
        _box(canvas, 10 + j(), 44 + j(), 5, 5, (1.0, 0.8, 0.0))        # right warning
        # Diagonal warning stripe across road
        for i in range(0, 16, 3):
            canvas[:, 8+i:10+i, 24+i:40+i] = 0.95
    elif scenario == "debris_on_road":
        # Large irregular debris field covering most of road with high contrast
        canvas[:, 18:48, 22:42] = 0.0   # dark base
        for _ in range(14):
            _box(canvas, rng.randint(20, 46), rng.randint(24, 40), 4, 4,
                 (rng.uniform(0.6, 0.9), rng.uniform(0.3, 0.5), rng.uniform(0.1, 0.3)))
        # bright hazard markers
        _box(canvas, 16 + j(), 22 + j(), 6, 4, (1.0, 0.5, 0.0))
        _box(canvas, 16 + j(), 40 + j(), 6, 4, (1.0, 0.5, 0.0))
    elif scenario == "animal_crossing":
        # Large animal (deer-like) plus multiple smaller ones, crossing whole road
        _box(canvas, 28 + j(), 32 + j(), 14, 18, (0.3, 0.55, 0.2))    # large green-brown
        _box(canvas, 40 + j(), 22 + j(), 7, 9, (0.35, 0.50, 0.18))
        _box(canvas, 20 + j(), 40 + j(), 6, 8, (0.32, 0.52, 0.19))
        # Animal trail marks
        for i in range(3):
            canvas[:, 48+i*2:50+i*2, 26+i*3:29+i*3] = 0.45
    elif scenario == "flooded_road":
        # Deep blue flood with ripple lines and floating debris
        canvas[:, 16:52, 22:42] = 0.0
        canvas[2, 16:52, 22:42] = 0.85    # bright blue channel
        canvas[0, 16:52, 22:42] = 0.05
        canvas[1, 16:52, 22:42] = 0.25
        # ripple lines (horizontal bands)
        for y in range(20, 50, 5):
            canvas[2, y:y+1, 22:42] = 0.55
            canvas[1, y:y+1, 22:42] = 0.60
        # Floating debris
        for _ in range(5):
            _box(canvas, rng.randint(20, 46), rng.randint(24, 40), 3, 5,
                 (0.6, 0.45, 0.2))
        # Warning signs at road entry
        _box(canvas, 14 + j(), 20 + j(), 7, 7, (1.0, 0.7, 0.0))
        _box(canvas, 14 + j(), 44 + j(), 7, 7, (1.0, 0.7, 0.0))
    elif scenario == "fallen_cyclist":
        # Horizontal figure PLUS bicycle shape PLUS concerned bystanders
        _box(canvas, 34 + j(), 30 + j(), 5, 18, (0.95, 0.55, 0.1))    # cyclist body
        _box(canvas, 30 + j(), 26 + j(), 8, 3, (0.6, 0.6, 0.6))       # wheel 1
        _box(canvas, 30 + j(), 38 + j(), 8, 3, (0.6, 0.6, 0.6))       # wheel 2
        _box(canvas, 42 + j(), 20 + j(), 6, 4, (0.8, 0.65, 0.55))     # bystander 1
        _box(canvas, 42 + j(), 44 + j(), 6, 4, (0.8, 0.65, 0.55))     # bystander 2
        # Emergency indicator
        _box(canvas, 50 + j(), 32 + j(), 5, 5, (1.0, 0.0, 0.0))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # mild noise for realism
    canvas = (canvas + 0.03 * torch.randn_like(canvas)).clamp(0, 1)
    return canvas


def generate(out_dir: str, n_per_common: int = 120, n_per_rare: int = 30,
             seed: int = 0) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    def build(scenarios: dict, n_each: int, split: str):
        images, actions, scen = [], [], []
        for name, action in scenarios.items():
            for _ in range(n_each):
                images.append(render_scene(name, rng))
                actions.append(ACTION_TO_IDX[action])
                scen.append(name)
        data = {
            "images": torch.stack(images),
            "actions": torch.tensor(actions, dtype=torch.long),
            "scenarios": scen,
        }
        torch.save(data, out / f"{split}.pt")
        print(f"  {split}: {len(images)} scenes "
              f"({len(scenarios)} scenarios x {n_each})")

    print(f"Generating synthetic driving scenes in {out} ...")
    build(COMMON_SCENARIOS, n_per_common, "common")
    build(RARE_SCENARIOS, n_per_rare, "rare")
    print("Done.")


def _smoke_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp, n_per_common=5, n_per_rare=3)
        d = torch.load(Path(tmp) / "common.pt", weights_only=False)
        print(f"common images: {tuple(d['images'].shape)}")
        print(f"unique actions: {sorted(set(d['actions'].tolist()))}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--n-common", type=int, default=120)
    p.add_argument("--n-rare", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.smoke_test:
        _smoke_test()
    elif args.generate:
        generate(args.out, args.n_common, args.n_rare, args.seed)


if __name__ == "__main__":
    main()