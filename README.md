# deltabn

``deltabn`` is an implementation of multi-state DAG structure learning for count data. The core model estimates:

- a shared backbone matrix across states
- state-specific deviations from that backbone
- per-state weighted adjacency matrices

This repository currently exposes the method as a Python module plus a command-line script. For reuse, the examples below show how to call the method from both Python and R and how to extract binary adjacency matrices (`Adj`).

## Repository contents

- `bbd_dagma_nb.py`: Core model and helper functions.
- `run_bbd_dagma_nb.py`: CLI for simulation and raw text input.
- `stability_selection.py`: Stability selection functions (usage in tutorial).
- `tutorial.ipynb`: Tutorial notebook.

## Installation

Use Python 3.10+.

```bash
pip install numpy pandas torch scikit-learn matplotlib igraph
```

If you want to call the method from R, also install `reticulate`.

```r
install.packages("reticulate")
```

## Python

```python
import numpy as np
from bbd_dagma_nb import MultiStateBackboneDAGMA_NB, numpy_to_torch_states

# Example: two states, d = 3 genes
X0 = np.array([[1, 0, 2], [0, 1, 1], [2, 1, 0]], dtype=np.float32)
X1 = np.array([[0, 2, 1], [1, 1, 0], [3, 0, 1]], dtype=np.float32)
X_states = [X0, X1]

device = "cpu"
thresh = 0.3

X_torch = numpy_to_torch_states(X_states, device=device)
model = MultiStateBackboneDAGMA_NB(d=X0.shape[1], n_states=len(X_states), device=device).to(device)
model.fit(
    X_torch,
    max_iter=1200,
    lr=3e-3,
    lam_l1=5e-3,
    lam_delta=1e-2,
    lam_group=5e-3,
    verbose=False,
)

# Weighted matrices
W_backbone = model.W0.detach().cpu().numpy()
W_states = [model.W_state(k).detach().cpu().numpy() for k in range(len(X_states))]

# Binary adjacency per state
Adj_states = [(np.abs(W) > thresh).astype(int) for W in W_states]
for A in Adj_states:
    np.fill_diagonal(A, 0)

```

## R

This repository does not yet provide a native R package. The recommended R interface is `reticulate`, which calls the Python module directly.

```r
library(reticulate)

use_python(Sys.which("python"), required = TRUE)

np <- import("numpy")
delta <- import_from_path("bbd_dagma_nb", path = ".")

X0 <- np$array(matrix(c(
  1, 0, 2,
  0, 1, 1,
  2, 1, 0
), nrow = 3, byrow = TRUE), dtype = "float32")

X1 <- np$array(matrix(c(
  0, 2, 1,
  1, 1, 0,
  3, 0, 1
), nrow = 3, byrow = TRUE), dtype = "float32")

X_states <- list(X0, X1)
device <- "cpu"
thresh <- 0.3

X_torch <- delta$numpy_to_torch_states(X_states, device = device)
model <- delta$MultiStateBackboneDAGMA_NB(
  d = as.integer(3),
  n_states = as.integer(length(X_states)),
  device = device
)$to(device)

model$fit(
  X_torch,
  max_iter = as.integer(1200),
  lr = 3e-3,
  lam_l1 = 5e-3,
  lam_delta = 1e-2,
  lam_group = 5e-3,
  verbose = FALSE
)

W0 <- py_to_r(model$W0$detach()$cpu()$numpy())
W_state0 <- py_to_r(model$W_state(as.integer(0))$detach()$cpu()$numpy())
W_state1 <- py_to_r(model$W_state(as.integer(1))$detach()$cpu()$numpy())

Adj_state0 <- ifelse(abs(W_state0) > thresh, 1, 0)
diag(Adj_state0) <- 0

Adj_state1 <- ifelse(abs(W_state1) > thresh, 1, 0)
diag(Adj_state1) <- 0
```

