import argparse
import json
from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bbd_dagma_nb import (
    MultiStateBackboneDAGMA_NB,
    consensus_backbone_from_states,
    edge_metrics,
    edge_score_metrics,
    edge_metrics_binary,
    generate_multistate_nb,
    is_dag_from_weight,
    numpy_to_torch_states,
)


def _plot_heatmap_true_est(true_B: np.ndarray, est_W: np.ndarray, title: str, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    vmax = max(float(np.max(np.abs(true_B))), float(np.max(np.abs(est_W))), 1.0)
    im0 = axes[0].imshow(true_B, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("True")
    axes[1].imshow(est_W, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_title("Estimated (weight)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.85)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_backbone_true_with_est_series(
    backbone_true: np.ndarray,
    backbone_est: np.ndarray,
    states_est: list,
    backbone_auprc: float,
    states_auprc: list,
    out_png: Path,
):
    mats = [backbone_true, backbone_est] + list(states_est)
    titles = ["True Backbone", f"Est Backbone\n(auprc: {backbone_auprc:.3f})"] + [
        f"Est State {i}\n(auprc: {states_auprc[i]:.3f})" for i in range(len(states_est))
    ]
    ncols = len(mats)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 2.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    vmax = max([float(np.max(np.abs(m))) for m in mats] + [1.0])
    for ax, m, t in zip(axes, mats, titles):
        im = ax.imshow(m, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.75)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_backbone_true_with_est_series_network(
    backbone_true: np.ndarray,
    backbone_est: np.ndarray,
    states_est: list,
    backbone_auprc: float,
    states_auprc: list,
    thresh: float,
    out_png: Path,
):
    A_true = (backbone_true != 0).astype(int)
    A_backbone = (np.abs(backbone_est) > thresh).astype(int)
    A_states = [(np.abs(w) > thresh).astype(int) for w in states_est]
    np.fill_diagonal(A_true, 0)
    np.fill_diagonal(A_backbone, 0)
    for A in A_states:
        np.fill_diagonal(A, 0)

    graphs = [_graph_from_adj01(A_true), _graph_from_adj01(A_backbone)] + [_graph_from_adj01(A) for A in A_states]
    titles = ["True Backbone", f"Est Backbone\n(auprc: {backbone_auprc:.3f})"] + [
        f"Est State {i}\n(auprc: {states_auprc[i]:.3f})" for i in range(len(A_states))
    ]

    # Shared layout for direct structural comparison.
    layout = graphs[0].layout_fruchterman_reingold()
    coords = [tuple(p) for p in layout]
    for g in graphs:
        g.vs["x"] = [c[0] for c in coords]
        g.vs["y"] = [c[1] for c in coords]

    ncols = len(graphs)
    fig, axes = plt.subplots(1, ncols, figsize=(2.9 * ncols, 3.0), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for ax, g, t in zip(axes, graphs, titles):
        ax.set_title(t, fontsize=9)
        ax.set_axis_off()
        ig.plot(
            g,
            target=ax,
            layout=list(zip(g.vs["x"], g.vs["y"])),
            vertex_size=12,
            vertex_color="lightgray",
            vertex_frame_width=0.5,
            edge_color="gray",
            bbox=(360, 360),
        )
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _graph_from_adj01(A01: np.ndarray) -> ig.Graph:
    g = ig.Graph.Adjacency(A01.astype(int).tolist(), mode=ig.ADJ_DIRECTED)
    g.vs["name"] = [str(i) for i in range(A01.shape[0])]
    return g


def _plot_network_true_est(true_B: np.ndarray, est_W: np.ndarray, thresh: float, title: str, out_png: Path):
    A_true = (true_B != 0).astype(int)
    A_est = (np.abs(est_W) > thresh).astype(int)
    np.fill_diagonal(A_true, 0)
    np.fill_diagonal(A_est, 0)

    g_true = _graph_from_adj01(A_true)
    g_est = _graph_from_adj01(A_est)
    layout = g_true.layout_fruchterman_reingold()
    coords = [tuple(p) for p in layout]
    g_true.vs["x"] = [c[0] for c in coords]
    g_true.vs["y"] = [c[1] for c in coords]
    g_est.vs["x"] = [c[0] for c in coords]
    g_est.vs["y"] = [c[1] for c in coords]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].set_title("True")
    axes[1].set_title(f"Estimated (|w|>{thresh:g})")
    for ax, g in zip(axes, [g_true, g_est]):
        ax.set_axis_off()
        ig.plot(
            g,
            target=ax,
            layout=list(zip(g.vs["x"], g.vs["y"])),
            vertex_size=14,
            vertex_color="lightgray",
            vertex_frame_width=0.5,
            edge_color="gray",
            bbox=(420, 420),
        )
    fig.suptitle(title)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_fused_pairs_indices(spec: str):
    if spec is None or str(spec).strip() == "":
        return None
    pairs = []
    toks = [t.strip() for t in spec.split(",") if t.strip()]
    for tok in toks:
        if ":" not in tok:
            raise ValueError("Each fused pair must be in i:j format")
        a, b = [x.strip() for x in tok.split(":", 1)]
        pairs.append((int(a), int(b)))
    return pairs


def resolve_raw_inputs(spec: str):
    if spec is None or str(spec).strip() == "":
        return None

    raw_path = Path(spec)
    if raw_path.exists() and raw_path.is_dir():
        files = sorted(
            [
                p
                for p in raw_path.iterdir()
                if p.is_file() and p.suffix.lower() in {".txt", ".tsv", ".csv"}
            ]
        )
        if len(files) == 0:
            raise ValueError(f"No raw text files found under: {raw_path}")
        return files

    if "," in spec:
        files = [Path(tok.strip()) for tok in spec.split(",") if tok.strip()]
        if len(files) == 0:
            raise ValueError("No raw text files were provided")
        return files

    return [raw_path]


def load_raw_txt_states(raw_files):
    X_states = []
    labels = []
    gene_cols = None

    for raw_file in raw_files:
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw txt file not found: {raw_file}")

        df = pd.read_csv(raw_file, sep=None, engine="python", index_col=0)
        if df.shape[1] == 0:
            raise ValueError(f"Raw txt file has no gene columns: {raw_file}")
        if df.isnull().any().any():
            raise ValueError(f"Raw txt file contains non-numeric or missing values: {raw_file}")

        cols = list(df.columns)
        if gene_cols is None:
            gene_cols = cols
        elif cols != gene_cols:
            raise ValueError(f"Gene column mismatch in {raw_file}")

        X_states.append(df.to_numpy(dtype=np.float32))
        labels.append(raw_file.stem)

    return labels, gene_cols, X_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--edge_prob", type=float, default=0.12)
    ap.add_argument("--graph_type", type=str, default="er", choices=["er", "ba"])
    ap.add_argument("--ba_m", type=int, default=2, help="BA attachment parameter when graph_type=ba")
    ap.add_argument("--delta_edges", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam_l1", type=float, default=5e-3)
    ap.add_argument("--lam_delta", type=float, default=1e-2)
    ap.add_argument("--lam_group", type=float, default=5e-3)
    ap.add_argument("--lam_fused", type=float, default=0.0, help="ordered-state fused penalty on adjacent Delta")
    ap.add_argument("--fused_pairs", type=str, default=None, help="comma-separated i:j pairs; overrides implicit chain")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default="results_bbd_dagma_nb")
    ap.add_argument("--consensus_k", type=int, default=None, help="k for k-of-S state consensus backbone")
    ap.add_argument("--save_plots", action="store_true", help="save heatmaps and network plots (true vs estimated)")
    ap.add_argument(
        "--raw_txt",
        type=str,
        default=None,
        help="raw txt/tsv/csv input. Use a directory, a comma-separated list of files, or a single file",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raw_files = resolve_raw_inputs(args.raw_txt)
    if raw_files is None:
        data, _ = generate_multistate_nb(
            d=args.d,
            n_states=args.states,
            n_per_state=args.n,
            edge_prob=args.edge_prob,
            delta_edges=args.delta_edges,
            seed=args.seed,
            graph_type=args.graph_type,
            ba_m=args.ba_m,
        )
        X_states = numpy_to_torch_states(data.X_states, device=args.device)
        data_labels = [f"state{k}" for k in range(args.states)]
        gene_cols = None
    else:
        labels, gene_cols, X_np_states = load_raw_txt_states(raw_files)
        X_states = numpy_to_torch_states(X_np_states, device=args.device)
        d = int(X_states[0].shape[1])
        args.d = d
        args.states = len(X_states)
        data = None
        data_labels = labels

    fused_pairs = parse_fused_pairs_indices(args.fused_pairs)

    model = MultiStateBackboneDAGMA_NB(d=args.d, n_states=args.states, device=args.device).to(args.device)
    model.fit(
        X_states,
        max_iter=args.epochs,
        lr=args.lr,
        lam_l1=args.lam_l1,
        lam_delta=args.lam_delta,
        lam_group=args.lam_group,
        lam_fused=args.lam_fused,
        fused_pairs=fused_pairs,
        verbose=True,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    W0 = model.W0.detach().cpu().numpy()
    Ws = [model.W_state(k).detach().cpu().numpy() for k in range(args.states)]

    k_majority = args.consensus_k if args.consensus_k is not None else ((args.states // 2) + 1)
    k_majority = max(1, min(args.states, k_majority))
    B_consensus_and = consensus_backbone_from_states(Ws, thresh=args.thresh, k=args.states)
    B_consensus_k = consensus_backbone_from_states(Ws, thresh=args.thresh, k=k_majority)

    metrics = {
        "raw_txt": [str(p) for p in raw_files] if raw_files is not None else None,
        "state_labels": data_labels,
        "backbone": None,
        "backbone_score": None,
        "backbone_is_dag": bool(is_dag_from_weight(W0, thresh=args.thresh)),
        "consensus_backbone_and": None,
        "consensus_backbone_and_is_dag": bool(is_dag_from_weight(B_consensus_and, thresh=0.5)),
        "consensus_backbone_k": None,
        "consensus_backbone_k_value": int(k_majority),
        "consensus_backbone_k_is_dag": bool(is_dag_from_weight(B_consensus_k, thresh=0.5)),
        "states": {},
    }

    if raw_files is None:
        metrics["backbone"] = edge_metrics(data.B_backbone, W0, thresh=args.thresh)
        metrics["backbone_score"] = edge_score_metrics(data.B_backbone, W0)
        metrics["consensus_backbone_and"] = edge_metrics_binary(data.B_backbone, B_consensus_and)
        metrics["consensus_backbone_k"] = edge_metrics_binary(data.B_backbone, B_consensus_k)

        for k in range(args.states):
            m = edge_metrics(data.B_states[k], Ws[k], thresh=args.thresh)
            m["auprc"] = edge_score_metrics(data.B_states[k], Ws[k])["auprc"]
            m["is_dag"] = bool(is_dag_from_weight(Ws[k], thresh=args.thresh))
            metrics["states"][str(k)] = m
    else:
        for k in range(args.states):
            m = {"auprc": None, "is_dag": bool(is_dag_from_weight(Ws[k], thresh=args.thresh))}
            metrics["states"][str(k)] = m

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if raw_files is not None:
        pd.DataFrame({"gene": gene_cols}).to_csv(outdir / "genes_used.csv", index=False)
        pd.DataFrame({"raw_txt": [str(p) for p in raw_files]}).to_csv(outdir / "raw_inputs.csv", index=False)

    np.save(outdir / "W_backbone_est.npy", W0)
    np.save(outdir / "B_backbone_consensus_and.npy", B_consensus_and)
    np.save(outdir / f"B_backbone_consensus_k{k_majority}.npy", B_consensus_k)
    for k in range(args.states):
        np.save(outdir / f"W_state{k}_est.npy", Ws[k])
        if raw_files is None:
            np.save(outdir / f"B_state{k}_true.npy", data.B_states[k])
            np.save(outdir / f"X_state{k}.npy", data.X_states[k])

    if raw_files is None and args.save_plots:
        bb_f1 = metrics["backbone"]["f1"]
        bb_auprc = metrics["backbone_score"]["auprc"]
        _plot_heatmap_true_est(
            data.B_backbone,
            W0,
            title=f"Backbone: True vs Estimated (Est backbone f1: {bb_f1:.3f}, Est backbone auprc: {bb_auprc:.3f})",
            out_png=outdir / "heatmap_backbone_true_vs_est.png",
        )
        _plot_network_true_est(
            data.B_backbone,
            W0,
            thresh=args.thresh,
            title=f"Backbone Network: True vs Estimated (Est backbone f1: {bb_f1:.3f}, Est backbone auprc: {bb_auprc:.3f})",
            out_png=outdir / "network_backbone_true_vs_est.png",
        )
        for k in range(args.states):
            st_f1 = metrics["states"][str(k)]["f1"]
            st_auprc = metrics["states"][str(k)]["auprc"]
            _plot_heatmap_true_est(
                data.B_states[k],
                Ws[k],
                title=f"State {k}: True vs Estimated (F1={st_f1:.3f}, AUPRC={st_auprc:.3f})",
                out_png=outdir / f"heatmap_state{k}_true_vs_est.png",
            )
            _plot_network_true_est(
                data.B_states[k],
                Ws[k],
                thresh=args.thresh,
                title=f"State {k} Network: True vs Estimated (F1={st_f1:.3f}, AUPRC={st_auprc:.3f})",
                out_png=outdir / f"network_state{k}_true_vs_est.png",
            )
        _plot_backbone_true_with_est_series(
            data.B_backbone,
            W0,
            Ws,
            backbone_auprc=bb_auprc,
            states_auprc=[metrics["states"][str(i)]["auprc"] for i in range(args.states)],
            out_png=outdir / "heatmap_backbone_true_plus_est_series.png",
        )
        _plot_backbone_true_with_est_series_network(
            data.B_backbone,
            W0,
            Ws,
            backbone_auprc=bb_auprc,
            states_auprc=[metrics["states"][str(i)]["auprc"] for i in range(args.states)],
            thresh=args.thresh,
            out_png=outdir / "network_backbone_true_plus_est_series.png",
        )
    elif raw_files is not None and args.save_plots:
        print("Skipping true-vs-est plots because raw txt mode does not provide ground-truth edges.")

    print("Saved:", outdir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
