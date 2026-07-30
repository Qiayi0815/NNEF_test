#!/usr/bin/env python3
"""Run native-basin molecular dynamics on your own PDB structure with the
NNEF energy function.

Starts the chain AT your input structure and runs overdamped Langevin
dynamics (Brownian dynamics: x(t+dt) = x(t) - alpha*grad(E) + beta*noise,
see nnef/physics/dynamics.py) under the chosen checkpoint's energy function.
Reports RMSD-to-start and radius of gyration over the trajectory -- a stable
model should stay close to the starting structure (a "flat plateau"); a
model with a poorly-shaped energy landscape will drift away.

Same checkpoint architecture table as score_for_collaborator.py (imported
from _collaborator_common, not duplicated) -- see that script's docstring
for why these flags are hard-coded rather than exposed as CLI options.

Usage:
    python nnef/scripts/run_md_for_collaborator.py \\
        --checkpoint yang --input my_protein.pdb --out_dir md_out/

    # longer run, more samples logged
    python nnef/scripts/run_md_for_collaborator.py \\
        --checkpoint ribbon --input my_protein.pdb --out_dir md_out/ \\
        --steps 100000 --log_interval 200

Runtime (measured on CPU, single core-bound run, an 80-residue chain --
throughput scales with chain length, not much with step count):
  ~14 steps/s  ->  2,000 steps (default) ~2.5 min, 20,000 steps ~24 min,
  100,000 steps ~2 hours. A GPU (--device cuda) is much faster; this
  project's own 500,000-step runs (~10 CPU-hours) were all run on GPU.
  Start with the default to sanity-check the setup, then raise --steps for
  a more decisive answer.

Outputs (under --out_dir):
    <name>_energy_rmsd.csv   step, energy, rmsd_to_start, rg  (one row per
                              logged step)
    <name>_trajectory.pdb    multi-MODEL PDB of the logged frames (Cbeta
                              trace, labelled "CA" -- same convention as
                              this project's other trajectory outputs), for
                              visualizing in PyMOL/ChimeraX or animating

--lr/--t_noise default to 3e-4/3e-4 -- this project's own validated
"landscape probing" regime (paired step size and noise). The Dynamics
class's own raw default, 3e-2, is sampler-DOMINATED (chains drift purely
from thermal noise, not the energy landscape) and is not a meaningful test
of the model -- don't raise these casually.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NNEF_DIR = os.path.abspath(os.path.join(_HERE, '..'))
_REPO_ROOT = os.path.abspath(os.path.join(_NNEF_DIR, '..'))
if _NNEF_DIR not in sys.path:
    sys.path.insert(0, _NNEF_DIR)

import options  # noqa: E402
from protein_os import Protein  # noqa: E402
from utils import load_protein_bead, test_setup, write_pdb_sample2  # noqa: E402
from physics.dynamics import Dynamics  # noqa: E402
from _collaborator_common import pdb_to_bead_df, resolve_args  # noqa: E402


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotate+translate P onto Q (both (N,3)); returns aligned P."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Pc @ R.T + Q.mean(0)


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))


def radius_of_gyration(coords: np.ndarray) -> float:
    c = coords - coords.mean(0)
    return float(np.sqrt((c ** 2).sum(axis=1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input', required=True, help='a single .pdb file')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--steps', type=int, default=2000,
                     help='MD steps (default 2000 -- ~2.5 min on CPU for a ~100-residue '
                          'chain, a quick sanity check; raise for a more decisive answer '
                          '-- 20000 ~24 min, 100000 ~2h on CPU; this project used 500000 '
                          'for its own long-run checks, on GPU)')
    ap.add_argument('--log_interval', type=int, default=50,
                     help='record a frame every N steps')
    ap.add_argument('--lr', type=float, default=3e-4, help='step size alpha')
    ap.add_argument('--t_noise', type=float, default=3e-4, help='noise scale beta')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    ns, arch = resolve_args(args.checkpoint, _REPO_ROOT, options)
    ns.device = args.device
    if arch.get('use_esm'):
        raise SystemExit(
            "[run_md_for_collaborator] ERROR: --checkpoint esm needs a precomputed "
            "ESM-C embedding for your sequence, which this script does not build. "
            "Running it anyway would silently run MD with the ESM branch switched "
            "off. See README.md section 5.")

    print(f'[run_md_for_collaborator] checkpoint={args.checkpoint}  '
          f'angle_dist={arch["angle_dist"]}  legacy_frame={arch["legacy_local_frame"]}  '
          f'steps={args.steps}  lr={args.lr}  t_noise={args.t_noise}  device={args.device}')
    device, _model, energy_fn, _pb = test_setup(ns)

    df, n_dropped = pdb_to_bead_df(args.input)
    if df is None:
        raise SystemExit(f'[run_md_for_collaborator] {args.input}: no residue with a '
                          f'complete backbone (N/CA/C[/CB]) found')
    if n_dropped:
        print(f'  warning: dropped {n_dropped} residue(s) missing a backbone atom; '
              f'running on the remaining {len(df)}')
    tmp_csv = args.input + '.__bead_tmp.csv'
    df.to_csv(tmp_csv, index=False)
    try:
        seq, coords, profile = load_protein_bead(tmp_csv, mode='CB', device=device)
    finally:
        os.remove(tmp_csv)

    protein = Protein(seq, coords.clone(), profile)
    start_coords = coords.detach().cpu().numpy().copy()

    dyn = Dynamics(energy_fn, protein, lr=args.lr, t_noise=args.t_noise,
                    num_steps=args.steps, log_interval=args.log_interval)
    t0 = time.time()
    dyn.run()
    elapsed = time.time() - t0
    print(f'[run_md_for_collaborator] {args.steps} steps in {elapsed:.1f}s '
          f'({args.steps / max(elapsed, 1e-9):.1f} steps/s)')

    name = os.path.splitext(os.path.basename(args.input))[0]
    rows = []
    pdb_frames = []
    for i, (coords_t, e) in enumerate(zip(dyn.sample, dyn.sample_energy)):
        c = coords_t.numpy()
        rows.append({
            'step': i * args.log_interval,
            'energy': e,
            'rmsd_to_start': rmsd(kabsch_align(c, start_coords), start_coords),
            'rg': radius_of_gyration(c),
        })
        pdb_frames.append(c)

    out_csv = os.path.join(args.out_dir, f'{name}_energy_rmsd.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    write_pdb_sample2(seq, pdb_frames, name, 'trajectory', args.out_dir)

    final = rows[-1]
    print(f'\nfinal (step {final["step"]}): RMSD to start = {final["rmsd_to_start"]:.2f} A, '
          f'Rg = {final["rg"]:.2f} A, energy = {final["energy"]:.2f}')
    print('A stable/well-shaped energy landscape holds RMSD roughly flat (thermal '
          'fluctuation around start); steady climb = the chain is drifting away from '
          'the input structure under this checkpoint.')
    print(f'\nwrote {out_csv}')
    print(f'wrote {os.path.join(args.out_dir, f"{name}_trajectory.pdb")}')


if __name__ == '__main__':
    main()
