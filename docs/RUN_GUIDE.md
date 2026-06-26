# Run and publish guide — SafeVLA-Retrieval

Complete sequence from a fresh clone to results, figures, and a published repo,
tuned for your RTX 3050 (4 GB). This project is light: the vision backbone is
frozen, so training the projection head takes only a few minutes.

---

## Step 1: Environment

```bash
# inside the project folder
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA 12.1 (works on your CUDA 12.2 driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# the rest
pip install -r requirements.txt

# verify GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 2: Smoke tests (no data, ~1 minute total)

Run each and confirm it prints sensible output before going further:

```bash
python3 encoder.py        # prints trainable vs frozen param counts, embedding shape
python3 memory.py         # prints OOD scores: known << novel, retrieval path [False, True]
python3 dataset.py --smoke-test
python3 train.py --smoke-test
```

The `train.py` smoke test runs the full pipeline (generate tiny data, train a few
epochs, build memory, evaluate) on synthetic data. If it prints a results block
with accuracies and an OOD AUROC, everything works.

---

## Step 3: Generate the benchmark

```bash
python3 dataset.py --generate --out data --n-common 120 --n-rare 30
```

This writes `data/common.pt` and `data/rare.pt`. Takes under a minute (it is
just rendering small schematic images). The `data/` folder is git-ignored.

---

## Step 4: Train and evaluate

```bash
python3 train.py --data data --epochs 30 --out-dir runs/exp1
```

You will see the contrastive loss decreasing each epoch, then a results block:

```
=== Results ===
  In-distribution (common) VLA acc:   0.9xx
  In-distribution (common) safe acc:  0.9xx
  Rare-scenario VLA acc:              0.xx   <- low; the failure mode
  Rare-scenario safe acc:             0.xx   <- higher; the fix
  OOD AUROC (common vs rare):         0.9xx
```

The key story: rare-scenario accuracy should rise from the VLA-only number to
the retrieval-augmented number, while common accuracy stays about the same.
The checkpoint and `results.json` are written to `runs/exp1/`.

Training the projection head on 4 GB uses well under 2 GB, so you will not hit
memory limits. If you ever do, lower `--batch-size` to 16.

---

## Step 5: Generate figures

```bash
python3 visualize.py --checkpoint runs/exp1/checkpoint.pt --data data
```

Produces in `docs/figures/`:

- `scenario_gallery.png` — common vs rare example scenes
- `embedding_tsne.png` — embedding space coloured by action, rare scenes overlaid
- `ood_histogram.png` — OOD score separation
- `accuracy_comparison.png` — the headline bar chart
- `retrieval_example.png` — a rare query and its retrieved neighbours

You can also generate just the gallery without a checkpoint:
```bash
python3 visualize.py --gallery-only --data data
```

---

## Step 6: Fill in the README numbers

Open `README.md`, find the results table with "*fill after run*", and paste the
real numbers from the training output (or read them from `runs/exp1/results.json`):

```bash
python3 -c "
import json
r = json.load(open('runs/exp1/results.json'))['results']
print(f'Common VLA:  {r[\"common_vla_acc\"]:.3f}')
print(f'Common safe: {r[\"common_safe_acc\"]:.3f}')
print(f'Rare VLA:    {r[\"rare_vla_acc\"]:.3f}')
print(f'Rare safe:   {r[\"rare_safe_acc\"]:.3f}')
print(f'OOD AUROC:   {r[\"ood_auroc\"]:.3f}')
"
```

---

## Step 7: Publish to GitHub

```bash
# create the repo on github.com first (Public, no README/license/gitignore),
# named SafeVLA-Retrieval, then:

git init -b main
git add .
git commit -m "Initial commit: retrieval-augmented rare-scenario robustness for VLA driving"
git remote add origin git@github.com:IQRAASGHAR1999/SafeVLA-Retrieval.git
git push -u origin main
```

Then on the repo page, add the About section:

- Description: `Rare-scenario robustness for Vision-Language-Action driving via retrieval-augmented action selection. PyTorch.`
- Topics: `computer-vision` `autonomous-driving` `vision-language-action` `out-of-distribution-detection` `retrieval-augmented` `pytorch` `contrastive-learning` `robustness` `multimodal`

After you have figures committed:

```bash
git add docs/figures/ README.md
git commit -m "Add results and figures"
git push
```

---

## Step 8: Pin it and update your profile

Pin SafeVLA-Retrieval on your GitHub profile. Suggested pinned set for this
application (CVI2 / autonomous driving):

1. SafeVLA-Retrieval (this project — most relevant to the position)
2. DynGS-Pro (your KSEM paper)
3. DIEP-ThermoSim (physics + uncertainty, shows range)
4. InterpretCV (interpretability — relevant to trustworthy autonomy)
5. MedSeg-Uncertainty (uncertainty quantification)
6. one more of your choice

Add a row to your profile README's featured-projects table:

```markdown
| **SafeVLA-Retrieval** | Rare-scenario robustness for Vision-Language-Action driving via retrieval-augmented action selection | 🚧 In progress |
```

---

## Common issues

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `--batch-size 16` (won't happen on this project, backbone is frozen) |
| `torch.cuda.is_available()` is False | reinstall torch with the cu121 index URL |
| t-SNE is slow | normal for the full set; it runs once in visualize.py, ~30s |
| Rare accuracy not improving | increase `--n-common` so the memory bank covers more cases, or lower `--ood-threshold` |
| sklearn missing | `pip install scikit-learn` |
