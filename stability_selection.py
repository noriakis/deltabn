from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bbd_dagma_nb import MultiStateBackboneDAGMA_NB


@dataclass
class StateData:
    manifest: pd.DataFrame
    labels: list[str]
    genes: list[str]
    x_states: list[torch.Tensor]
    cell_names: list[list[str]]


@dataclass
class SplitData:
    train_idx: list[np.ndarray]
    val_idx: list[np.ndarray]
    train_x: list[torch.Tensor]
    val_x: list[torch.Tensor]
    train_offsets: list[torch.Tensor] | None
    val_offsets: list[torch.Tensor] | None


def parse_float_grid(text: str) -> list[float]:
    if isinstance(text, (list, tuple, np.ndarray)):
        vals = [float(x) for x in text]
        if not vals:
            raise ValueError("Expected at least one numeric value")
        return vals
    vals = [float(tok) for tok in re.split(r"[,\s]+", str(text).strip()) if tok]
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


def parse_penalty_grids(
    grids: Any,
    lambda_ref_grid: Any,
    lambda_delta_grid: Any,
) -> tuple[list[float], list[float], list[tuple[float, float]]]:
    if grids is None:
        lam_refs = parse_float_grid(lambda_ref_grid)
        lam_deltas = parse_float_grid(lambda_delta_grid)
        return lam_refs, lam_deltas, list(itertools.product(lam_refs, lam_deltas))

    if isinstance(grids, dict):
        ref_grid = (
            grids.get("lambda_ref_grid")
            if "lambda_ref_grid" in grids
            else grids.get("lam_ref_grid", grids.get("lambda_ref", grids.get("lam_ref")))
        )
        delta_grid = (
            grids.get("lambda_delta_grid")
            if "lambda_delta_grid" in grids
            else grids.get("lam_delta_grid", grids.get("lambda_delta", grids.get("lam_delta")))
        )
        if ref_grid is None or delta_grid is None:
            raise ValueError("grids dict must include lambda_ref/lam_ref and lambda_delta/lam_delta")
        lam_refs = parse_float_grid(ref_grid)
        lam_deltas = parse_float_grid(delta_grid)
        return lam_refs, lam_deltas, list(itertools.product(lam_refs, lam_deltas))

    if not isinstance(grids, (list, tuple)) or len(grids) == 0:
        raise ValueError("grids must be a non-empty dict or list")

    combos: list[tuple[float, float]] = []
    for item in grids:
        if isinstance(item, dict):
            if "lam_ref" in item:
                lam_ref = item["lam_ref"]
            elif "lambda_ref" in item:
                lam_ref = item["lambda_ref"]
            else:
                raise ValueError("grid dict entries must include lam_ref or lambda_ref")

            if "lam_delta" in item:
                lam_delta = item["lam_delta"]
            elif "lambda_delta" in item:
                lam_delta = item["lambda_delta"]
            else:
                raise ValueError("grid dict entries must include lam_delta or lambda_delta")
            combos.append((float(lam_ref), float(lam_delta)))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            combos.append((float(item[0]), float(item[1])))
        else:
            raise ValueError("grid entries must be (lam_ref, lam_delta) pairs or dicts")

    lam_refs = sorted({x[0] for x in combos})
    lam_deltas = sorted({x[1] for x in combos})
    return lam_refs, lam_deltas, combos


def state_data_from_arrays(
    x_states: list[Any] | tuple[Any, ...],
    labels: list[str] | None = None,
    genes: list[str] | None = None,
    cell_names: list[list[str]] | None = None,
) -> StateData:
    if x_states is None or len(x_states) == 0:
        raise ValueError("x_states must contain at least one state matrix")

    tensors: list[torch.Tensor] = []
    d: int | None = None
    for state_id, x in enumerate(x_states):
        if isinstance(x, torch.Tensor):
            xt = x.detach().to(dtype=torch.float32, device="cpu").clone()
        else:
            xt = torch.tensor(np.asarray(x), dtype=torch.float32)
        if xt.ndim != 2:
            raise ValueError(f"state {state_id} must be a 2D matrix; got shape {tuple(xt.shape)}")
        if d is None:
            d = int(xt.shape[1])
        elif int(xt.shape[1]) != d:
            raise ValueError(f"state {state_id} has {xt.shape[1]} genes, expected {d}")
        tensors.append(xt)

    n_states = len(tensors)
    if labels is None:
        labels = [f"state{k}" for k in range(n_states)]
    labels = [str(x) for x in labels]
    if len(labels) != n_states:
        raise ValueError("labels must have one entry per state")

    if genes is None:
        genes = [f"g{j}" for j in range(int(d or 0))]
    genes = [str(x) for x in genes]
    if len(genes) != int(d or 0):
        raise ValueError("genes must have one entry per matrix column")

    if cell_names is None:
        cell_names = [[f"{labels[k]}_cell{i}" for i in range(int(x.shape[0]))] for k, x in enumerate(tensors)]
    if len(cell_names) != n_states:
        raise ValueError("cell_names must have one list per state")
    for k, names in enumerate(cell_names):
        if len(names) != int(tensors[k].shape[0]):
            raise ValueError(f"cell_names[{k}] length must match state {k} rows")

    manifest = pd.DataFrame({"state": labels, "file": ["in_memory"] * n_states})
    return StateData(
        manifest=manifest,
        labels=labels,
        genes=genes,
        x_states=tensors,
        cell_names=[[str(x) for x in names] for names in cell_names],
    )


def _looks_like_state_matrices(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        return False
    first = obj[0]
    shape = getattr(first, "shape", None)
    return shape is not None and len(shape) == 2


def resolve_reference_state(reference_state: str | int, labels: list[str]) -> int:
    text = str(reference_state)
    if text in labels:
        return labels.index(text)
    try:
        idx = int(text)
    except ValueError as exc:
        raise ValueError(
            f"reference_state must be an integer index or exact state name; got {reference_state!r}"
        ) from exc
    if idx < 0 or idx >= len(labels):
        raise ValueError(f"reference_state index {idx} is out of range for {len(labels)} states")
    return idx


def read_state_count_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 0:
        raise ValueError(f"No columns found in {path}")

    first_col = str(df.columns[0])
    first_col_lc = first_col.lower()
    has_cell_id_col = first_col.startswith("Unnamed:") or first_col_lc in {
        "cell",
        "cell_id",
        "cell_name",
        "barcode",
    }
    if has_cell_id_col:
        df = df.set_index(df.columns[0])
    else:
        df.index = [f"{path.stem}_{i}" for i in range(df.shape[0])]
    return df


def load_manifest_states(indir: Path) -> StateData:
    manifest = pd.read_csv(indir / "manifest.csv")
    x_states: list[torch.Tensor] = []
    labels: list[str] = []
    cell_names: list[list[str]] = []
    gene_cols: list[str] | None = None

    for _, row in manifest.iterrows():
        path = indir / str(row["file"])
        df = read_state_count_table(path)
        if gene_cols is None:
            gene_cols = list(df.columns)
        elif list(df.columns) != gene_cols:
            raise ValueError(f"Gene columns mismatch in {path}")
        x_states.append(torch.tensor(df.values, dtype=torch.float32))
        labels.append(str(row["state"]))
        cell_names.append([str(x) for x in df.index])

    if gene_cols is None:
        raise ValueError(f"No states found in {indir / 'manifest.csv'}")
    return StateData(manifest=manifest, labels=labels, genes=gene_cols, x_states=x_states, cell_names=cell_names)


def load_metadata(path: Path | None, offset_mode: str) -> pd.DataFrame | None:
    if offset_mode != "ncount_rna":
        return None
    if path is None:
        raise ValueError("--metadata_tsv is required when --offset_mode ncount_rna")
    meta = pd.read_csv(path, sep="\t", index_col=0)
    if "nCount_RNA" not in meta.columns:
        raise ValueError("metadata must include nCount_RNA column for offset mode")
    meta.index = meta.index.astype(str)
    return meta


def split_indices(x_states: list[torch.Tensor], fraction: float, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    for state_id, x in enumerate(x_states):
        n = int(x.shape[0])
        if n < 2:
            raise ValueError(f"Need at least two cells in every state; state {state_id} has {n}")
        n_train = int(round(n * float(fraction)))
        n_train = min(max(1, n_train), n - 1)
        perm = rng.permutation(n)
        train_idx.append(np.sort(perm[:n_train]))
        val_idx.append(np.sort(perm[n_train:]))
    return train_idx, val_idx


def subset_states(x_states: list[torch.Tensor], indices: list[np.ndarray]) -> list[torch.Tensor]:
    return [x[torch.tensor(idx, dtype=torch.long)].clone() for x, idx in zip(x_states, indices)]


def rowsum_offsets_for_split(
    train_x: list[torch.Tensor],
    val_x: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    all_train = np.concatenate([x.sum(dim=1).cpu().numpy() for x in train_x])
    lib_center = float(np.median(all_train))
    if lib_center <= 0:
        raise ValueError("Invalid train median row-sum for offset computation")

    def make(xs: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for x in xs:
            totals = x.sum(dim=1).cpu().numpy()
            off = np.log(np.maximum(totals, 1.0) / lib_center)
            out.append(torch.tensor(off, dtype=torch.float32))
        return out

    return make(train_x), make(val_x)


def ncount_offsets_for_split(
    cell_names: list[list[str]],
    train_idx: list[np.ndarray],
    val_idx: list[np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    train_cells = [cell_names[k][int(i)] for k, idx in enumerate(train_idx) for i in idx]
    missing = sorted(set(train_cells).difference(metadata.index))
    if missing:
        raise ValueError(f"metadata missing {len(missing)} train cells; first missing cell: {missing[0]}")
    all_n = metadata.loc[train_cells, "nCount_RNA"].astype(float).values
    lib_center = float(np.median(all_n))
    if lib_center <= 0:
        raise ValueError("Invalid train median nCount_RNA for offset computation")

    def make(idx_by_state: list[np.ndarray]) -> list[torch.Tensor]:
        out = []
        for k, idx in enumerate(idx_by_state):
            cells = [cell_names[k][int(i)] for i in idx]
            missing_local = sorted(set(cells).difference(metadata.index))
            if missing_local:
                raise ValueError(f"metadata missing {len(missing_local)} cells; first missing cell: {missing_local[0]}")
            ncount = metadata.loc[cells, "nCount_RNA"].astype(float).values
            off = np.log(np.maximum(ncount, 1.0) / lib_center)
            out.append(torch.tensor(off, dtype=torch.float32))
        return out

    return make(train_idx), make(val_idx)


def offsets_for_split(
    offset_mode: str,
    train_x: list[torch.Tensor],
    val_x: list[torch.Tensor],
    cell_names: list[list[str]],
    train_idx: list[np.ndarray],
    val_idx: list[np.ndarray],
    metadata: pd.DataFrame | None,
) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
    if offset_mode == "none":
        return None, None
    if offset_mode == "rowsum":
        return rowsum_offsets_for_split(train_x, val_x)
    if metadata is None:
        raise ValueError("metadata is required for ncount_rna offsets")
    return ncount_offsets_for_split(cell_names, train_idx, val_idx, metadata)


def make_splits(
    data: StateData,
    n_refits: int,
    subsample_fraction: float,
    seed: int,
    offset_mode: str,
    metadata: pd.DataFrame | None,
) -> list[SplitData]:
    splits: list[SplitData] = []
    for r in range(n_refits):
        rng = np.random.default_rng(int(seed) + 7919 * (r + 1))
        train_idx, val_idx = split_indices(data.x_states, subsample_fraction, rng)
        train_x = subset_states(data.x_states, train_idx)
        val_x = subset_states(data.x_states, val_idx)
        train_offsets, val_offsets = offsets_for_split(
            offset_mode=offset_mode,
            train_x=train_x,
            val_x=val_x,
            cell_names=data.cell_names,
            train_idx=train_idx,
            val_idx=val_idx,
            metadata=metadata,
        )
        splits.append(
            SplitData(
                train_idx=train_idx,
                val_idx=val_idx,
                train_x=train_x,
                val_x=val_x,
                train_offsets=train_offsets,
                val_offsets=val_offsets,
            )
        )
    return splits


def set_torch_seed(seed: int, device: str) -> None:
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fit_one_model(
    *,
    d: int,
    s: int,
    model_seed: int,
    train_x: list[torch.Tensor],
    train_offsets: list[torch.Tensor] | None,
    ref_idx: int,
    lam_ref: float,
    lam_delta: float,
    args: argparse.Namespace,
) -> MultiStateBackboneDAGMA_NB:
    set_torch_seed(model_seed, args.device)
    model = MultiStateBackboneDAGMA_NB(d=d, n_states=s, device=args.device).to(args.device)
    model.fit_reference_delta(
        x_states=[x.to(args.device) for x in train_x],
        ref_idx=ref_idx,
        offsets=[o.to(args.device) for o in train_offsets] if train_offsets is not None else None,
        max_iter=args.epochs,
        lr=args.lr,
        lam_l1=args.lam_l1,
        lam_w0=float(lam_ref),
        lam_state=args.lam_state,
        lam_delta=float(lam_delta),
        lam_group=args.lam_group,
        dagma_s=args.dagma_s,
        mu0=args.mu0,
        mu_decay=args.mu_decay,
        mu_decay_per=args.mu_decay_per,
        verbose=args.verbose,
    )
    return model


@torch.no_grad()
def validation_loglik(
    model: MultiStateBackboneDAGMA_NB,
    val_x: list[torch.Tensor],
    val_offsets: list[torch.Tensor] | None,
    ref_idx: int,
) -> tuple[float, list[dict[str, float | int]]]:
    val_x = [x.to(model.W0.device) for x in val_x]
    offsets = [o.to(model.W0.device) for o in val_offsets] if val_offsets is not None else None
    per_state = []
    ll_sum = 0.0
    n_sum = 0
    for k, xk in enumerate(val_x):
        wk = model.W0 - torch.diag(torch.diag(model.W0)) if k == ref_idx else model.W_state(k)
        offk = offsets[k] if offsets is not None else None
        ll = float(model.nb_loglik(xk, wk, model.alpha[k], model.theta[k], offset=offk))
        n = int(xk.shape[0])
        per_state.append({"state_index": int(k), "n_cells": n, "validation_ll_per_cell": ll})
        ll_sum += ll * n
        n_sum += n
    return ll_sum / max(n_sum, 1), per_state


def mean_se(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0, 0.0
    sd = float(arr.std(ddof=1))
    return mean, sd, sd / math.sqrt(arr.size)


def differential_state_ids(n_states: int, ref_idx: int, include_reference: bool) -> list[int]:
    if include_reference:
        return list(range(n_states))
    return [k for k in range(n_states) if k != ref_idx]


def instability(selection_freq: np.ndarray, state_ids: list[int]) -> float:
    if not state_ids:
        return 0.0
    d = int(selection_freq.shape[-1])
    mask = ~np.eye(d, dtype=bool)
    vals = selection_freq[np.asarray(state_ids, dtype=int)][:, mask]
    if vals.size == 0:
        return 0.0
    return float(np.mean(2.0 * vals * (1.0 - vals)))


def select_by_one_se(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    best_idx = summary_df["validation_ll_mean"].astype(float).idxmax()
    best = summary_df.loc[best_idx].to_dict()
    best_mean = float(best["validation_ll_mean"])
    best_se = float(best.get("validation_ll_se", 0.0))
    if not np.isfinite(best_se):
        best_se = 0.0
    cutoff = best_mean - best_se

    out = summary_df.copy()
    out["within_one_se"] = out["validation_ll_mean"].astype(float) >= cutoff
    eligible = out.loc[out["within_one_se"]].copy()
    selected = eligible.sort_values(
        ["instability", "validation_ll_mean", "lam_ref", "lam_delta"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).iloc[0]
    out["selected"] = out["combo_id"] == selected["combo_id"]
    rule = {
        "best_combo_id": int(best["combo_id"]),
        "best_validation_ll_mean": best_mean,
        "best_validation_ll_se": best_se,
        "one_se_cutoff": float(cutoff),
        "selected_combo_id": int(selected["combo_id"]),
    }
    return out, rule


def select_by_validation_then_instability(
    summary_df: pd.DataFrame,
    instability_beta: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = summary_df.copy()
    beta = float(instability_beta)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("--instability_beta must be a finite non-negative value")

    best_by_ref_rows = []
    for lam_ref, g in out.groupby("lam_ref", sort=True):
        best = g.sort_values(
            ["validation_ll_mean", "lam_delta"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        best_by_ref_rows.append(
            {
                "lam_ref": float(lam_ref),
                "best_combo_id": int(best["combo_id"]),
                "best_lam_delta": float(best["lam_delta"]),
                "best_validation_ll_mean": float(best["validation_ll_mean"]),
                "best_validation_ll_se": float(best.get("validation_ll_se", 0.0)),
            }
        )

    best_by_ref = pd.DataFrame(best_by_ref_rows)
    ref_pick = best_by_ref.sort_values(
        ["best_validation_ll_mean", "lam_ref"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    selected_lam_ref = float(ref_pick["lam_ref"])

    ref_path = out.loc[out["lam_ref"].astype(float) == selected_lam_ref].copy()
    ref_path = ref_path.sort_values("lam_delta", ascending=True, kind="mergesort")
    stable_path = ref_path.loc[ref_path["instability"].astype(float) <= beta]
    stable_delta_found = bool(len(stable_path) > 0)
    if stable_delta_found:
        selected = stable_path.iloc[0]
        fallback_reason = None
    else:
        selected = ref_path.sort_values(
            ["instability", "lam_delta"],
            ascending=[True, True],
            kind="mergesort",
        ).iloc[0]
        fallback_reason = "no_lambda_delta_satisfied_instability_beta"

    selected_combo_id = int(selected["combo_id"])
    out["lambda_ref_selected"] = out["lam_ref"].astype(float) == selected_lam_ref
    out["instability_eligible"] = out["lambda_ref_selected"] & (out["instability"].astype(float) <= beta)
    out["selected_by_fallback"] = False
    if not stable_delta_found:
        out.loc[out["combo_id"] == selected_combo_id, "selected_by_fallback"] = True
    out["selected"] = out["combo_id"] == selected_combo_id

    rule = {
        "selection_rule": "validation_likelihood_then_first_stable_delta",
        "lambda_ref_selection": "choose lam_ref whose best validation_ll_mean over lam_delta is largest",
        "lambda_delta_selection": "within selected lam_ref, choose smallest lam_delta with instability <= instability_beta",
        "instability_beta": beta,
        "selected_lam_ref": selected_lam_ref,
        "selected_lam_delta": float(selected["lam_delta"]),
        "selected_combo_id": selected_combo_id,
        "stable_delta_found": stable_delta_found,
        "fallback_reason": fallback_reason,
        "lambda_ref_validation_path": best_by_ref_rows,
    }
    return out, rule


def make_selected_edge_table(
    selection_freq: np.ndarray,
    delta_mean: np.ndarray,
    delta_sd: np.ndarray,
    genes: list[str],
    labels: list[str],
    state_ids: list[int],
    max_rows: int,
    ) -> pd.DataFrame:
    d = len(genes)
    mask = ~np.eye(d, dtype=bool)
    src_idx, dst_idx = np.where(mask)
    gene_arr = np.asarray(genes, dtype=object)
    frames = []
    for k in state_ids:
        frames.append(
            pd.DataFrame(
                {
                    "state_index": int(k),
                    "state": labels[k],
                    "src_idx": src_idx,
                    "dst_idx": dst_idx,
                    "src_gene": gene_arr[src_idx],
                    "dst_gene": gene_arr[dst_idx],
                    "selection_frequency": selection_freq[k][mask],
                    "delta_mean": delta_mean[k][mask],
                    "delta_sd": delta_sd[k][mask],
                }
            )
        )
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df["abs_delta_mean"] = np.abs(df["delta_mean"].astype(float))
        df = df.sort_values(
            ["selection_frequency", "abs_delta_mean", "state_index", "src_idx", "dst_idx"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
        if max_rows > 0:
            df = df.head(max_rows)
    else:
        df = pd.DataFrame(
            columns=[
                "state_index",
                "state",
                "src_idx",
                "dst_idx",
                "src_gene",
                "dst_gene",
                "selection_frequency",
                "delta_mean",
                "delta_sd",
                "abs_delta_mean",
            ]
        )
    return df


def write_selected_edge_table(
    path: Path,
    selection_freq: np.ndarray,
    delta_mean: np.ndarray,
    delta_sd: np.ndarray,
    genes: list[str],
    labels: list[str],
    state_ids: list[int],
    max_rows: int,
) -> None:
    df = make_selected_edge_table(
        selection_freq=selection_freq,
        delta_mean=delta_mean,
        delta_sd=delta_sd,
        genes=genes,
        labels=labels,
        state_ids=state_ids,
        max_rows=max_rows,
    )
    df.to_csv(path, index=False)


def full_offsets(
    data: StateData,
    offset_mode: str,
    metadata: pd.DataFrame | None,
) -> list[torch.Tensor] | None:
    if offset_mode == "none":
        return None
    if offset_mode == "rowsum":
        train_offsets, _ = rowsum_offsets_for_split(data.x_states, data.x_states)
        return train_offsets
    if metadata is None:
        raise ValueError("metadata is required for ncount_rna offsets")
    all_cells = [cell for cells in data.cell_names for cell in cells]
    missing = sorted(set(all_cells).difference(metadata.index))
    if missing:
        raise ValueError(f"metadata missing {len(missing)} cells; first missing cell: {missing[0]}")
    lib_center = float(np.median(metadata.loc[all_cells, "nCount_RNA"].astype(float).values))
    if lib_center <= 0:
        raise ValueError("Invalid full-data median nCount_RNA for offset computation")
    offsets = []
    for cells in data.cell_names:
        ncount = metadata.loc[cells, "nCount_RNA"].astype(float).values
        offsets.append(torch.tensor(np.log(np.maximum(ncount, 1.0) / lib_center), dtype=torch.float32))
    return offsets


def refit_selected_full(
    *,
    data: StateData,
    ref_idx: int,
    lam_ref: float,
    lam_delta: float,
    args: argparse.Namespace,
    metadata: pd.DataFrame | None,
    outdir: Path,
) -> None:
    offsets = full_offsets(data, args.offset_mode, metadata)
    model = fit_one_model(
        d=len(data.genes),
        s=len(data.labels),
        model_seed=int(args.seed) + 900001,
        train_x=data.x_states,
        train_offsets=offsets,
        ref_idx=ref_idx,
        lam_ref=lam_ref,
        lam_delta=lam_delta,
        args=args,
    )
    w_ref = model.W0.detach().cpu().numpy()
    np.save(outdir / "W_reference_selected_full.npy", w_ref)
    np.save(outdir / "W_backbone_selected_full.npy", w_ref)
    for k in range(len(data.labels)):
        wk = w_ref.copy() if k == ref_idx else model.W_state(k).detach().cpu().numpy()
        dk = model.Delta[k].detach().cpu().numpy()
        np.save(outdir / f"W_state{k}_selected_full.npy", wk)
        np.save(outdir / f"Delta_state{k}_selected_full.npy", dk)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_arg_parser(require_indir: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=require_indir, default=None)
    ap.add_argument("--reference_state", type=str, default="0", help="reference state index or exact name")
    ap.add_argument("--lambda_ref_grid", "--lam_ref_grid", dest="lambda_ref_grid", type=str, default="0.002")
    ap.add_argument("--lambda_delta_grid", "--lam_delta_grid", dest="lambda_delta_grid", type=str, default="0.01")
    ap.add_argument("--n_refits", type=int, default=25, help="20-30 is the intended range")
    ap.add_argument("--subsample_fraction", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam_l1", type=float, default=0.002)
    ap.add_argument("--lam_state", type=float, default=None)
    ap.add_argument("--lam_group", type=float, default=0.005)
    ap.add_argument("--dagma_s", type=float, default=1.0)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--mu_decay", type=float, default=0.2)
    ap.add_argument("--mu_decay_per", type=int, default=500)
    ap.add_argument("--edge_thresh", "--thresh", dest="edge_thresh", type=float, default=0.25)
    ap.add_argument("--instability_beta", type=float, default=0.05)
    ap.add_argument("--offset_mode", type=str, default="rowsum", choices=["none", "rowsum", "ncount_rna"])
    ap.add_argument("--metadata_tsv", type=str, default=None)
    ap.add_argument("--include_reference_delta_in_instability", action="store_true")
    ap.add_argument("--max_edge_rows", type=int, default=50000)
    ap.add_argument("--refit_selected_full", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default="results_stability_selection")
    ap.add_argument("--verbose", action="store_true")
    return ap


def make_stability_selection_args(**kwargs: Any) -> argparse.Namespace:
    grids = kwargs.pop("grids", None)
    args = build_arg_parser(require_indir=False).parse_args([])
    valid_keys = set(vars(args))
    unknown = sorted(set(kwargs).difference(valid_keys))
    if unknown:
        raise TypeError(f"Unknown stability selection option(s): {', '.join(unknown)}")

    if "outdir" not in kwargs:
        args.outdir = None

    for key, value in kwargs.items():
        if key in {"indir", "metadata_tsv", "outdir"} and value is not None:
            value = str(value)
        setattr(args, key, value)

    args.grids = grids
    return args


def run_stability_selection(
    args: argparse.Namespace | dict[str, Any] | list[Any] | tuple[Any, ...] | None = None,
    *,
    x_states: list[Any] | tuple[Any, ...] | None = None,
    grids: Any = None,
    labels: list[str] | None = None,
    genes: list[str] | None = None,
    cell_names: list[list[str]] | None = None,
    data: StateData | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run stability selection from in-memory state matrices or CLI arguments."""
    if args is not None and _looks_like_state_matrices(args):
        if x_states is not None:
            raise TypeError("Pass state matrices either positionally or with x_states=, not both")
        x_states = args
        args = None

    if args is None:
        if grids is not None:
            kwargs["grids"] = grids
        args = make_stability_selection_args(**kwargs)
    elif kwargs:
        raise TypeError("Pass either an argparse.Namespace or keyword options, not both")
    elif isinstance(args, dict):
        if grids is not None:
            args = dict(args)
            args["grids"] = grids
        args = make_stability_selection_args(**args)
    elif not isinstance(args, argparse.Namespace):
        raise TypeError("args must be an argparse.Namespace, dict, or None")
    elif grids is not None:
        args.grids = grids

    if args.n_refits < 1:
        raise ValueError("--n_refits must be positive")
    if not (0.0 < float(args.subsample_fraction) < 1.0):
        raise ValueError("--subsample_fraction must be between 0 and 1")

    indir = Path(args.indir) if args.indir is not None else None
    outdir = Path(args.outdir) if args.outdir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    if data is not None:
        state_data = data
    elif x_states is not None:
        state_data = state_data_from_arrays(
            x_states=x_states,
            labels=labels,
            genes=genes,
            cell_names=cell_names,
        )
    elif indir is not None:
        state_data = load_manifest_states(indir)
    else:
        raise TypeError("run_stability_selection() requires X as the first argument, x_states=..., or indir=...")

    data = state_data
    ref_idx = resolve_reference_state(args.reference_state, data.labels)
    metadata = load_metadata(Path(args.metadata_tsv) if args.metadata_tsv else None, args.offset_mode)
    lam_refs, lam_deltas, combos = parse_penalty_grids(
        grids=getattr(args, "grids", None),
        lambda_ref_grid=args.lambda_ref_grid,
        lambda_delta_grid=args.lambda_delta_grid,
    )
    state_ids = differential_state_ids(
        n_states=len(data.labels),
        ref_idx=ref_idx,
        include_reference=bool(args.include_reference_delta_in_instability),
    )

    splits = make_splits(
        data=data,
        n_refits=args.n_refits,
        subsample_fraction=args.subsample_fraction,
        seed=args.seed,
        offset_mode=args.offset_mode,
        metadata=metadata,
    )

    d = len(data.genes)
    s = len(data.labels)
    offdiag = ~np.eye(d, dtype=bool)
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selection_freq_by_combo = []
    delta_mean_by_combo = []
    delta_sd_by_combo = []

    for combo_id, (lam_ref, lam_delta) in enumerate(combos):
        t_combo = time.perf_counter()
        val_lls: list[float] = []
        sel_count = np.zeros((s, d, d), dtype=np.uint16)
        delta_sum = np.zeros((s, d, d), dtype=np.float64)
        delta_sq_sum = np.zeros((s, d, d), dtype=np.float64)

        for refit_id, split in enumerate(splits):
            t_fit = time.perf_counter()
            model = fit_one_model(
                d=d,
                s=s,
                model_seed=int(args.seed) + 100003 * (refit_id + 1),
                train_x=split.train_x,
                train_offsets=split.train_offsets,
                ref_idx=ref_idx,
                lam_ref=lam_ref,
                lam_delta=lam_delta,
                args=args,
            )
            val_ll, per_state = validation_loglik(model, split.val_x, split.val_offsets, ref_idx=ref_idx)
            val_lls.append(val_ll)

            deltas = model.Delta.detach().cpu().numpy().astype(np.float64, copy=False)
            deltas[:, np.eye(d, dtype=bool)] = 0.0
            selected = np.abs(deltas) > float(args.edge_thresh)
            selected[:, ~offdiag] = False
            sel_count += selected.astype(np.uint16)
            delta_sum += deltas
            delta_sq_sum += deltas * deltas

            run_rows.append(
                {
                    "combo_id": combo_id,
                    "refit_id": refit_id,
                    "lam_ref": float(lam_ref),
                    "lam_delta": float(lam_delta),
                    "validation_ll": float(val_ll),
                    "fit_sec": float(time.perf_counter() - t_fit),
                    "train_cells_total": int(sum(len(x) for x in split.train_idx)),
                    "validation_cells_total": int(sum(len(x) for x in split.val_idx)),
                    "validation_ll_by_state_json": json.dumps(per_state),
                }
            )
            print(
                f"[combo {combo_id+1}/{len(combos)} refit {refit_id+1}/{len(splits)}] "
                f"lam_ref={lam_ref:g} lam_delta={lam_delta:g} val_ll={val_ll:.6f}",
                flush=True,
            )

        n = max(len(splits), 1)
        freq = sel_count.astype(np.float32) / float(n)
        delta_mean = delta_sum / float(n)
        delta_var = np.maximum(delta_sq_sum / float(n) - delta_mean * delta_mean, 0.0)
        delta_sd = np.sqrt(delta_var)
        val_mean, val_sd, val_se = mean_se(val_lls)
        instab = instability(freq, state_ids=state_ids)

        selection_freq_by_combo.append(freq.astype(np.float32))
        delta_mean_by_combo.append(delta_mean.astype(np.float32))
        delta_sd_by_combo.append(delta_sd.astype(np.float32))
        summary_rows.append(
            {
                "combo_id": combo_id,
                "lam_ref": float(lam_ref),
                "lam_delta": float(lam_delta),
                "n_refits": int(n),
                "validation_ll_mean": val_mean,
                "validation_ll_sd": val_sd,
                "validation_ll_se": val_se,
                "instability": instab,
                "mean_selected_delta_edges_per_refit": float(sel_count[np.asarray(state_ids, dtype=int)][:, offdiag].sum() / n)
                if state_ids
                else 0.0,
                "combo_sec": float(time.perf_counter() - t_combo),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df, selection_rule = select_by_validation_then_instability(
        summary_df,
        instability_beta=float(args.instability_beta),
    )
    selected_combo_id = int(selection_rule["selected_combo_id"])
    selected_row = summary_df.loc[summary_df["combo_id"] == selected_combo_id].iloc[0].to_dict()

    selection_arr = np.stack(selection_freq_by_combo, axis=0)
    delta_mean_arr = np.stack(delta_mean_by_combo, axis=0)
    delta_sd_arr = np.stack(delta_sd_by_combo, axis=0)

    run_df = pd.DataFrame(run_rows)
    selected_edges_df = make_selected_edge_table(
        selection_freq=selection_arr[selected_combo_id],
        delta_mean=delta_mean_arr[selected_combo_id],
        delta_sd=delta_sd_arr[selected_combo_id],
        genes=data.genes,
        labels=data.labels,
        state_ids=state_ids,
        max_rows=int(args.max_edge_rows),
    )

    if outdir is not None:
        np.save(outdir / "selection_frequency_by_combo.npy", selection_arr)
        np.save(outdir / "delta_mean_by_combo.npy", delta_mean_arr)
        np.save(outdir / "delta_sd_by_combo.npy", delta_sd_arr)
        np.save(outdir / "Delta_selection_freq_selected.npy", selection_arr[selected_combo_id])
        np.save(outdir / "Delta_mean_selected.npy", delta_mean_arr[selected_combo_id])
        np.save(outdir / "Delta_sd_selected.npy", delta_sd_arr[selected_combo_id])

        run_df.to_csv(outdir / "subsample_refit_runs.csv", index=False)
        summary_df.to_csv(outdir / "stability_summary.csv", index=False)
        pd.DataFrame({"gene": data.genes}).to_csv(outdir / "genes_used.csv", index=False)
        data.manifest.to_csv(outdir / "manifest.csv", index=False)
        selected_edges_df.to_csv(outdir / "selected_edge_frequencies.csv", index=False)

    if args.refit_selected_full and outdir is None:
        raise ValueError("refit_selected_full requires outdir=... when using the Python API")

    if args.refit_selected_full:
        refit_selected_full(
            data=data,
            ref_idx=ref_idx,
            lam_ref=float(selected_row["lam_ref"]),
            lam_delta=float(selected_row["lam_delta"]),
            args=args,
            metadata=metadata,
            outdir=outdir,
        )

    summary = {
        "indir": str(indir) if indir is not None else None,
        "source": "manifest" if indir is not None else "in_memory",
        "outdir": str(outdir) if outdir is not None else None,
        "n_states": int(s),
        "states": data.labels,
        "reference_index": int(ref_idx),
        "reference_state": data.labels[ref_idx],
        "n_genes": int(d),
        "n_cells_per_state": [int(x.shape[0]) for x in data.x_states],
        "lambda_ref_grid": [float(x) for x in lam_refs],
        "lambda_delta_grid": [float(x) for x in lam_deltas],
        "penalty_grid": [{"lam_ref": float(a), "lam_delta": float(b)} for a, b in combos],
        "n_refits": int(args.n_refits),
        "subsample_fraction": float(args.subsample_fraction),
        "edge_thresh": float(args.edge_thresh),
        "threshold_rule": "abs(Delta[k][i,j]) > edge_thresh",
        "instability_beta": float(args.instability_beta),
        "offset_mode": args.offset_mode,
        "instability_state_indices": state_ids,
        "instability_state_labels": [data.labels[k] for k in state_ids],
        "selection_rule": selection_rule,
        "selected_penalty": selected_row,
        "outputs": (
            {
                "stability_summary": "stability_summary.csv",
                "subsample_refit_runs": "subsample_refit_runs.csv",
                "selection_frequency_by_combo": "selection_frequency_by_combo.npy",
                "selected_edge_frequencies": "selected_edge_frequencies.csv",
            }
            if outdir is not None
            else {}
        ),
    }
    summary_ready = json_ready(summary)
    if outdir is not None:
        with open(outdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_ready, f, indent=2)
        with open(outdir / "selected_penalty.json", "w", encoding="utf-8") as f:
            json.dump(json_ready(selected_row), f, indent=2)

    if outdir is not None:
        print("Saved:", outdir, flush=True)

    result = dict(summary_ready)
    result.update(
        {
            "summary_df": summary_df,
            "subsample_refit_runs_df": run_df,
            "selected_edge_frequencies_df": selected_edges_df,
            "selection_frequency_by_combo": selection_arr,
            "delta_mean_by_combo": delta_mean_arr,
            "delta_sd_by_combo": delta_sd_arr,
            "delta_selection_freq_selected": selection_arr[selected_combo_id],
            "delta_mean_selected": delta_mean_arr[selected_combo_id],
            "delta_sd_selected": delta_sd_arr[selected_combo_id],
        }
    )
    return result


def main() -> None:
    args = build_arg_parser().parse_args()
    run_stability_selection(args)


if __name__ == "__main__":
    main()
