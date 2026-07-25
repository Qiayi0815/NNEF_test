# NNEF energy function — collaborator test kit

Score your own PDB structure(s) with a neural-network protein energy
function, in one command, no training data or cluster access needed.

This is a minimal, self-contained slice of an ongoing FYP project built on
top of Yang, Xiong & Zonta's NNEF (2022) — see [Background](#background)
below. It adds:

- a retrain on a larger (28,446-chain) dataset,
- a Ramachandran head + v2 coordinate frame (**ribbon**),
- a frozen ESM-C sequence-embedding adapter (**esm**, not yet wired into
  this script — see [§5](#5-the-esm-checkpoint-not-yet-usable-here)),
- an ablation replacing the Gaussian angle/radial output distributions with
  circular (von Mises) and directional (von Mises–Fisher) ones.

Six trained checkpoints are included (~8 MB each) — nothing to download.

## 1. Setup (~5 min)

```bash
conda create -n nnef-score python=3.10 -y
conda activate nnef-score
pip install -r requirements.txt
```

## 2. Try it on the included example

```bash
python nnef/scripts/score_for_collaborator.py \
    --checkpoint yang --input examples/1zzk.pdb --out_csv scores.csv
```

Expected: prints `[1zzk] energy = 807.12` and writes `scores.csv`.

## 3. Score your own structure(s)

```bash
python nnef/scripts/score_for_collaborator.py \
    --checkpoint yang \
    --input your_protein.pdb \
    --out_csv scores.csv
```

**Lower energy = more native-like**, by construction (the model is trained
so `P ∝ exp(-E)`, assigning low energy to real/near-native structures and
higher energy to decoys).

Score a whole folder of structures (e.g. several decoys of one target) in one
call:

```bash
python nnef/scripts/score_for_collaborator.py \
    --checkpoint ribbon \
    --input path/to/decoys/ \
    --out_csv scores.csv
```

If you have known quality labels per structure (e.g. GDT_TS, TM-score), pass
them and the script also reports the correlation — the standard way this
project validates a checkpoint:

```bash
# labels.csv:  structure,label
python nnef/scripts/score_for_collaborator.py \
    --checkpoint yang --input path/to/decoys/ --out_csv scores.csv \
    --labels_csv labels.csv
```

A working checkpoint should show **energy and quality label negatively
correlated** (Pearson r around −0.8 to −0.9 is what these checkpoints get on
the CASP14 T1026 benchmark, energy vs. GDT_TS across 512 decoys).

## 4. Which checkpoint?

| `--checkpoint` | What it is | Notes |
|---|---|---|
| `yang` | Yang 2022 architecture, retrained on 28k chains, Gaussian output | closest to the original paper; start here |
| `yang_vonmises` | same backbone, circular von Mises distribution for angles | ablation — see caveat below |
| `yang_vmf` | same backbone, von Mises–Fisher (spherical) direction distribution | ablation — see caveat below |
| `ribbon` | `yang` + v2 coordinate frame + Ramachandran head | our main modified model |
| `ribbon_vonmises` | `ribbon` + von Mises angles | ablation |
| `esm` | `ribbon` + frozen ESM-C 600M embedding | not usable via this script yet, see §5 |

All six were trained on the same 28,446-chain dataset, 1000 epochs, and are
directly comparable to each other.

**Caveat on the `*_vonmises`/`*_vmf` checkpoints:** in our own testing, they
fit the training distribution marginally better but were **less stable in
long molecular-dynamics runs** than the Gaussian baseline at the same sampler
settings — plausibly because each output distribution implies a different
"effective temperature" for the energy landscape, and everything was run at
one fixed sampler setting rather than recalibrating it per checkpoint. Decoy
correlation (what this script computes) was **not** noticeably different
across the ablation, so this caveat mainly matters if you go on to run
MD/sampling with these checkpoints — not for plain structure scoring.

## 4b. Input format details

- Standard PDB format, one file per structure.
- Every residue needs backbone **N, CA, C** (+ **CB**, except glycine, which
  uses CA). Residues missing any of these are silently dropped — the script
  prints a warning with the count; check it matches what you expect.
- Multi-chain files: all chains are read and scored together as one sequence
  (matches how the model was trained on single continuous chains — split the
  PDB first if you want chains scored independently).
- No MSA / evolutionary profile needed — every checkpoint here reads raw
  sequence directly (`--seq_type residue`), not a profile.

## 5. The `esm` checkpoint (not yet usable here)

The `esm` checkpoint needs a precomputed per-residue ESM-C 600M embedding for
your sequence. `score_for_collaborator.py` does not compute this on the fly,
and **deliberately refuses to run `--checkpoint esm`** rather than silently
scoring with the ESM branch switched off (which would produce a
plausible-looking but wrong number — the checkpoint's ESM weights would
contribute nothing to the energy).

If you want to test this checkpoint, get in touch — we'll extend the script
rather than have you hand-build the CLI call (the ESM flags are easy to get
subtly wrong).

## Questions / issues

This is active research code from an ongoing project, not a polished
package — if something breaks or a result looks off, that's useful signal,
not necessarily your mistake. Several checkpoint/architecture mismatches in
this project have historically been the kind of bug that fails *silently*
(wrong energy, no crash), so if a number looks surprising, flag it rather
than assuming it's expected.

## Background

Code derived from Yang, Xiong & Zonta, *"Construction of a neural network
energy function for protein physics,"* [biorxiv
2021](https://www.biorxiv.org/content/10.1101/2021.04.26.441401v1). The
energy function is a probability density model, `P(x) ∝ exp(-E(x))`, learned
from local 3D protein structure. See `LICENSE` (MIT, original copyright
Huan Yang).
