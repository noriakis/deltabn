import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import igraph as ig
except ImportError:
    ig = None
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import average_precision_score


@dataclass
class SimulationData:
    X_states: List[np.ndarray]
    B_backbone: np.ndarray
    B_states: List[np.ndarray]
    order: np.ndarray


def _offdiag_mask(d: int, device=None) -> torch.Tensor:
    return ~torch.eye(d, dtype=torch.bool, device=device)


def random_order_dag(
    d: int,
    edge_prob: float,
    rng: np.random.Generator,
    graph_type: str = "er",
    ba_m: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    if ig is None:
        raise ImportError("igraph is required for random graph simulation utilities")
    graph_type = graph_type.lower()
    # Keep deterministic seed behavior while using igraph's internal RNG path.
    random.seed(int(rng.integers(0, 2**31 - 1)))

    if graph_type == "er":
        g = ig.Graph.Erdos_Renyi(n=d, p=edge_prob, directed=False, loops=False)
    elif graph_type == "ba":
        m = int(max(1, min(ba_m, d - 1)))
        g = ig.Graph.Barabasi(n=d, m=m)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}. Use 'er' or 'ba'.")

    # ZICO-compatible DAG conversion path.
    g.to_directed(mode="acyclic")
    B = np.array(g.get_adjacency().data, dtype=np.float32)
    np.fill_diagonal(B, 0.0)

    # Build a valid topological order from resulting DAG, then define allowed
    # edge directions for delta toggling so perturbation remains acyclic.
    order = np.array(g.topological_sorting(mode="out"), dtype=int)
    rank = np.empty(d, dtype=int)
    rank[order] = np.arange(d)
    allowed = [(i, j) for i in range(d) for j in range(d) if rank[i] < rank[j]]
    return B, order, allowed


def perturb_backbone(
    B0: np.ndarray,
    n_states: int,
    delta_edges: int,
    allowed: List[Tuple[int, int]],
    rng: np.random.Generator,
    delta_mode: str = "toggle",
) -> List[np.ndarray]:
    delta_mode = str(delta_mode).lower()
    if delta_mode not in {"toggle", "add_only"}:
        raise ValueError(f"Unsupported delta_mode: {delta_mode}. Use 'toggle' or 'add_only'.")

    states = []
    edge_set = set(allowed)
    for _ in range(n_states):
        Bs = B0.copy()
        if delta_edges > 0:
            if delta_mode == "toggle":
                idx = rng.choice(len(allowed), size=min(delta_edges, len(allowed)), replace=False)
                for k in idx:
                    i, j = allowed[k]
                    Bs[i, j] = 1.0 - Bs[i, j]
            else:
                # Keep backbone edges shared and add state-specific edges only from missing edges.
                missing = [(i, j) for (i, j) in allowed if B0[i, j] == 0]
                if len(missing) > 0:
                    idx = rng.choice(len(missing), size=min(delta_edges, len(missing)), replace=False)
                    for k in idx:
                        i, j = missing[k]
                        Bs[i, j] = 1.0
        # Ensure at least one edge to avoid degenerate all-zero state.
        if Bs.sum() == 0 and len(edge_set) > 0:
            i, j = allowed[rng.integers(0, len(allowed))]
            Bs[i, j] = 1.0
        states.append(Bs)
    return states


def draw_weights(
    B: np.ndarray,
    rng: np.random.Generator,
    low: float = 0.2,
    high: float = 0.9,
) -> np.ndarray:
    signs = rng.choice([-1.0, 1.0], size=B.shape)
    mags = rng.uniform(low, high, size=B.shape)
    W = signs * mags * B
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def sample_nb_sem(
    W: np.ndarray,
    order: np.ndarray,
    n: int,
    rng: np.random.Generator,
    intercept_sd: float = 0.3,
    log_mu_clip: float = 3.0,
    theta: float = 8.0,
) -> np.ndarray:
    d = W.shape[0]
    alpha = rng.normal(0.0, intercept_sd, size=d)
    X = np.zeros((n, d), dtype=np.float32)

    for j in order:
        parents = np.where(W[:, j] != 0)[0]
        if len(parents) > 0:
            log_mu = alpha[j] + X[:, parents] @ W[parents, j]
        else:
            log_mu = np.full(n, alpha[j], dtype=np.float32)

        mu = np.exp(np.minimum(log_mu, log_mu_clip))
        p = theta / (theta + mu)
        xj = rng.negative_binomial(theta, p).astype(np.float32)
        X[:, j] = xj
    return X


def generate_multistate_nb(
    d: int,
    n_states: int,
    n_per_state: int,
    edge_prob: float,
    delta_edges: int,
    seed: int,
    graph_type: str = "er",
    ba_m: int = 2,
    delta_mode: str = "toggle",
) -> Tuple[SimulationData, List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    B0, order, allowed = random_order_dag(d=d, edge_prob=edge_prob, rng=rng, graph_type=graph_type, ba_m=ba_m)
    B_states = perturb_backbone(
        B0=B0,
        n_states=n_states,
        delta_edges=delta_edges,
        allowed=allowed,
        rng=rng,
        delta_mode=delta_mode,
    )

    W_states = [draw_weights(Bs, rng) for Bs in B_states]
    X_states = [sample_nb_sem(W=W, order=order, n=n_per_state, rng=rng) for W in W_states]

    data = SimulationData(X_states=X_states, B_backbone=B0, B_states=B_states, order=order)
    return data, W_states


class MultiStateBackboneDAGMA_NB(nn.Module):
    def __init__(self, d: int, n_states: int, init_scale: float = 0.05, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.s = n_states
        rnd = lambda *shape: torch.randn(*shape, device=device) * init_scale

        self.W0 = nn.Parameter(rnd(d, d))
        self.Delta = nn.Parameter(rnd(n_states, d, d))
        self.alpha = nn.Parameter(torch.zeros(n_states, d, device=device))
        self.theta = nn.Parameter(torch.ones(n_states, d, device=device))
        self._zero_diag_()

    def _zero_diag_(self):
        with torch.no_grad():
            self.W0.fill_diagonal_(0.0)
            for k in range(self.s):
                self.Delta[k].fill_diagonal_(0.0)

    def W_state(self, k: int) -> torch.Tensor:
        Wk = self.W0 + self.Delta[k]
        return Wk - torch.diag(torch.diag(Wk))

    def acyclicity_logdet(self, W: torch.Tensor, s: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        d = W.shape[0]
        S = W * W
        A = s * torch.eye(d, device=W.device) - S
        _, logabsdet = torch.linalg.slogdet(A)
        return -logabsdet + d * math.log(s + eps)

    def nb_loglik(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        alpha: torch.Tensor,
        theta: torch.Tensor,
        offset: torch.Tensor = None,
    ) -> torch.Tensor:
        eta = alpha.view(1, -1) + X @ W
        if offset is not None:
            eta = eta + offset.view(-1, 1)
        eta = eta.clamp(-10.0, 8.0)
        mu = torch.exp(eta)
        r = F.softplus(theta).clamp(1e-4, 1e3).view(1, -1)

        log_r = torch.log(r)
        log_rpm = torch.log(r + mu + 1e-12)
        log_p = log_r - log_rpm
        log_1mp = torch.log(mu + 1e-12) - log_rpm

        k = X
        ll = torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1.0) + r * log_p + k * log_1mp
        return ll.sum(dim=1).mean()

    def group_penalty(self) -> torch.Tensor:
        Ws = torch.stack([self.W_state(k) for k in range(self.s)], dim=0)
        g = torch.sqrt((Ws * Ws).sum(dim=0) + 1e-8)
        mask = _offdiag_mask(self.d, device=Ws.device)
        return g[mask].sum()

    def fused_delta_penalty(
        self,
        state_order: List[int] = None,
        fused_pairs: List[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.s <= 1:
            return torch.tensor(0.0, device=self.W0.device)

        if fused_pairs is None:
            order = list(range(self.s)) if state_order is None else list(state_order)
            if len(order) != self.s or len(set(order)) != self.s:
                raise ValueError("state_order must be a permutation of state indices")
            if min(order) < 0 or max(order) >= self.s:
                raise ValueError("state_order contains out-of-range indices")
            pairs = list(zip(order[:-1], order[1:]))
        else:
            pairs = [(int(a), int(b)) for a, b in fused_pairs]
            if len(pairs) == 0:
                return torch.tensor(0.0, device=self.W0.device)
            for a, b in pairs:
                if a < 0 or b < 0 or a >= self.s or b >= self.s:
                    raise ValueError("fused_pairs contains out-of-range indices")
                if a == b:
                    raise ValueError("fused_pairs cannot contain self-pairs")

        mask = _offdiag_mask(self.d, device=self.W0.device)
        fused = torch.tensor(0.0, device=self.W0.device)
        for a, b in pairs:
            fused = fused + (self.Delta[b] - self.Delta[a]).abs()[mask].sum()
        return fused

    def fit(
        self,
        X_states: List[torch.Tensor],
        offsets: List[torch.Tensor] = None,
        max_iter: int = 2000,
        lr: float = 3e-3,
        lam_l1: float = 5e-3,
        lam_w0: float = None,
        lam_state: float = None,
        lam_delta: float = 1e-2,
        lam_center: float = 0.0,
        lam_group: float = 5e-3,
        lam_fused: float = 0.0,
        state_order: List[int] = None,
        fused_pairs: List[Tuple[int, int]] = None,
        dagma_s: float = 1.0,
        mu0: float = 1.0,
        mu_decay: float = 0.2,
        mu_decay_per: int = 500,
        verbose: bool = True,
    ):
        X_states = [x.to(self.W0.device) for x in X_states]
        if offsets is not None:
            offsets = [o.to(self.W0.device) for o in offsets]
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99))
        lam_w0_eff = lam_l1 if lam_w0 is None else lam_w0
        lam_state_eff = lam_l1 if lam_state is None else lam_state

        mu = mu0
        for t in range(max_iter):
            opt.zero_grad()

            nll = 0.0
            h = 0.0
            l1_state = 0.0
            for k, X in enumerate(X_states):
                Wk = self.W_state(k)
                offk = offsets[k] if offsets is not None else None
                nll = nll - self.nb_loglik(X, Wk, self.alpha[k], self.theta[k], offset=offk)
                h = h + self.acyclicity_logdet(Wk, s=dagma_s)
                l1_state = l1_state + Wk.abs().sum()

            fused = self.fused_delta_penalty(state_order=state_order, fused_pairs=fused_pairs)
            center = self.Delta.sum(dim=0).abs().sum()
            reg = (
                lam_w0_eff * self.W0.abs().sum()
                + lam_state_eff * l1_state
                + lam_delta * self.Delta.abs().sum()
                + lam_center * center
                + lam_group * self.group_penalty()
                + lam_fused * fused
            )
            loss = mu * (nll + reg) + h
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            self._zero_diag_()

            if (t + 1) % mu_decay_per == 0:
                mu *= mu_decay

            if verbose and ((t + 1) % max(50, max_iter // 20) == 0 or t == 0):
                print(
                    f"iter={t+1} loss={loss.item():.3f} nll={float(nll):.3f} "
                    f"h={float(h):.3f} reg={float(reg):.3f} mu={mu:.4f}"
                )

        return self


def edge_metrics(B_true: np.ndarray, W_est: np.ndarray, thresh: float = 0.3) -> Dict[str, float]:
    Bp = (np.abs(W_est) > thresh).astype(np.float32)
    Bt = (B_true != 0).astype(np.float32)
    d = Bt.shape[0]
    mask = ~np.eye(d, dtype=bool)
    yp = Bp[mask]
    yt = Bt[mask]

    tp = float((yp * yt).sum())
    fp = float((yp * (1.0 - yt)).sum())
    fn = float(((1.0 - yp) * yt).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    fdr = fp / (tp + fp + 1e-8)
    shd = float(np.abs(yp - yt).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdr": fdr,
        "shd": shd,
    }


def edge_score_metrics(B_true: np.ndarray, W_est: np.ndarray) -> Dict[str, float]:
    Bt = (B_true != 0).astype(np.float32)
    d = Bt.shape[0]
    mask = ~np.eye(d, dtype=bool)
    y_true = Bt[mask].ravel()
    y_score = np.abs(W_est[mask].ravel())
    auprc = float(average_precision_score(y_true, y_score))
    return {"auprc": auprc}


def edge_metrics_binary(B_true: np.ndarray, B_est: np.ndarray) -> Dict[str, float]:
    Bt = (B_true != 0).astype(np.float32)
    Be = (B_est != 0).astype(np.float32)
    d = Bt.shape[0]
    mask = ~np.eye(d, dtype=bool)
    yp = Be[mask]
    yt = Bt[mask]

    tp = float((yp * yt).sum())
    fp = float((yp * (1.0 - yt)).sum())
    fn = float(((1.0 - yp) * yt).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    fdr = fp / (tp + fp + 1e-8)
    shd = float(np.abs(yp - yt).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdr": fdr,
        "shd": shd,
    }


def consensus_backbone_from_states(
    Ws: List[np.ndarray],
    thresh: float,
    k: int,
) -> np.ndarray:
    """
    Build k-of-S consensus binary adjacency from state-level weighted matrices.
    """
    A = np.stack([(np.abs(W) > thresh).astype(np.int32) for W in Ws], axis=0)
    votes = A.sum(axis=0)
    B = (votes >= k).astype(np.int32)
    np.fill_diagonal(B, 0)
    return B


def numpy_to_torch_states(X_states: List[np.ndarray], device: str) -> List[torch.Tensor]:
    return [torch.tensor(X, dtype=torch.float32, device=device) for X in X_states]


def is_dag_from_weight(W: np.ndarray, thresh: float = 0.3) -> bool:
    """
    Check DAG property on thresholded adjacency inferred from weighted matrix.
    """
    A = (np.abs(W) > thresh).astype(np.int32)
    np.fill_diagonal(A, 0)
    d = A.shape[0]

    indeg = A.sum(axis=0).astype(np.int64)
    queue = [i for i in range(d) if indeg[i] == 0]
    visited = 0

    while queue:
        u = queue.pop()
        visited += 1
        children = np.where(A[u] != 0)[0]
        for v in children:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    return visited == d
