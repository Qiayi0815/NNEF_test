"""Shared checkpoint architecture table + PDB parsing for the collaborator-
facing scripts (score_for_collaborator.py, run_md_for_collaborator.py).

Kept in ONE place deliberately: every checkpoint/architecture mismatch bug
hit during this project happened because two call sites disagreed about a
model's flags. Both collaborator scripts import CHECKPOINTS from here so
there is exactly one place that can be wrong.
"""
from __future__ import annotations

import os

import pandas as pd

_BASE_ARCH = dict(
    seq_len=14, seq_type='residue', residue_type_num=20,
    embed_size=32, dim=128, n_layers=4, attn_heads=4,
    mixture_r=2, mixture_angle=3,
    smooth_gaussian=True, smooth_r=0.3, smooth_angle=45,
    coords_angle_loss_lamda=1, profile_loss_lamda=10, coords_rama_loss_lamda=1,
    use_position_weights=True, cen_seg_loss_lamda=1, oth_seg_loss_lamda=3,
)

CHECKPOINTS = {
    'yang': dict(
        load_exp='checkpoints/yang', legacy_local_frame=True,
        mixture_seq=1, mixture_rama=0, angle_dist='gaussian', r_dist='gaussian',
    ),
    'yang_vonmises': dict(
        load_exp='checkpoints/yang_vonmises', legacy_local_frame=True,
        mixture_seq=1, mixture_rama=0, angle_dist='vonmises', r_dist='gaussian',
    ),
    'yang_vmf': dict(
        load_exp='checkpoints/yang_vmf', legacy_local_frame=True,
        mixture_seq=1, mixture_rama=0, angle_dist='vmf', r_dist='gaussian',
    ),
    'ribbon': dict(
        load_exp='checkpoints/ribbon', legacy_local_frame=False,
        mixture_seq=1, mixture_rama=10, angle_dist='gaussian', r_dist='gaussian',
    ),
    'ribbon_vonmises': dict(
        load_exp='checkpoints/ribbon_vonmises', legacy_local_frame=False,
        mixture_seq=1, mixture_rama=10, angle_dist='vonmises', r_dist='gaussian',
    ),
    'esm': dict(
        load_exp='checkpoints/esm', legacy_local_frame=False,
        mixture_seq=1, mixture_rama=10, angle_dist='gaussian', r_dist='gaussian',
        use_esm=True, esm_dim_in=1152, esm_dim_out=32, esm_pool='per_residue',
        # esm_h5_path is NOT set here -- needs a per-residue ESM-C embedding
        # for YOUR sequence, which neither collaborator script builds. Both
        # scripts refuse to run --checkpoint esm rather than silently scoring
        # with the ESM branch switched off. See README.md section 5.
    ),
}

_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def pdb_to_bead_df(pdb_path: str):
    """One PDB -> one bead DataFrame (chain_id, group_num, group_name, N/CA/C/CB
    xyz). Mirrors nnef/data_prep_scripts/fetch_and_beads.py's extract_beads()
    exactly, so this matches the training-data format bit-for-bit. Returns
    (None, n_dropped) if no residue has a complete backbone (+CB)."""
    from Bio.PDB import PDBParser, Selection

    structure = PDBParser(QUIET=True).get_structure(os.path.basename(pdb_path), pdb_path)
    cols = ['chain_id', 'group_num', 'group_name',
            'xn', 'yn', 'zn', 'xca', 'yca', 'zca', 'xc', 'yc', 'zc', 'xcb', 'ycb', 'zcb']
    bead = {c: [] for c in cols}
    n_dropped = 0
    for res in Selection.unfold_entities(structure, 'R'):
        if res.id[0] != ' ':
            continue  # skip waters / heteroatoms
        resname = res.get_resname().upper()
        if resname not in _THREE_TO_ONE:
            continue
        try:
            n, ca, c = (res[a].get_coord() for a in ('N', 'CA', 'C'))
        except KeyError:
            n_dropped += 1
            continue
        if resname == 'GLY':
            cb = ca
        else:
            try:
                cb = res['CB'].get_coord()
            except KeyError:
                n_dropped += 1
                continue
        bead['chain_id'].append(res.parent.id)
        bead['group_num'].append(res.id[1])
        bead['group_name'].append(_THREE_TO_ONE[resname])
        bead['xn'].append(n[0]); bead['yn'].append(n[1]); bead['zn'].append(n[2])
        bead['xca'].append(ca[0]); bead['yca'].append(ca[1]); bead['zca'].append(ca[2])
        bead['xc'].append(c[0]); bead['yc'].append(c[1]); bead['zc'].append(c[2])
        bead['xcb'].append(cb[0]); bead['ycb'].append(cb[1]); bead['zcb'].append(cb[2])
    if not bead['chain_id']:
        return None, n_dropped
    return pd.DataFrame(bead), n_dropped


def resolve_args(checkpoint_key: str, repo_root: str, options_module):
    """Build the argparse.Namespace for test_setup(), with per-checkpoint
    architecture baked in. Shared by both collaborator scripts."""
    import argparse

    if checkpoint_key not in CHECKPOINTS:
        raise SystemExit(f'--checkpoint must be one of {sorted(CHECKPOINTS)}, got {checkpoint_key!r}')
    arch = CHECKPOINTS[checkpoint_key]
    parser = options_module.get_fold_parser()
    ns = argparse.Namespace(**vars(parser.parse_args([])))
    for k, v in _BASE_ARCH.items():
        setattr(ns, k, v)
    for k, v in arch.items():
        setattr(ns, k, v)
    ns.load_exp = os.path.join(repo_root, arch['load_exp'])
    if not os.path.isfile(os.path.join(ns.load_exp, 'models', 'model.pt')):
        raise SystemExit(
            f"ERROR: no checkpoint at {ns.load_exp}/models/model.pt -- "
            f"did you pull/clone the checkpoints?")
    return ns, arch
