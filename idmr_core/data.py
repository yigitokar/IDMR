from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TextData:
    """
    Minimal wrapper around (C, V, M) so callers don’t have to pass raw arrays.
    """

    C: np.ndarray  # shape (n, d)
    V: np.ndarray  # shape (n, p)
    M: np.ndarray  # shape (n,)

    @classmethod
    def from_arrays(cls, C: np.ndarray, V: np.ndarray, M: Optional[np.ndarray] = None) -> "TextData":
        if M is None:
            M = C.sum(axis=1)
        # Basic shape checks; keep lightweight to avoid surprises later.
        if C.shape[0] != V.shape[0]:
            raise ValueError(f"C and V must have same n; got {C.shape[0]} vs {V.shape[0]}")
        if M.shape[0] != C.shape[0]:
            raise ValueError(f"M length {M.shape[0]} does not match C rows {C.shape[0]}")
        return cls(C=np.asarray(C), V=np.asarray(V), M=np.asarray(M))

    @property
    def n(self) -> int:
        return self.C.shape[0]

    @property
    def d(self) -> int:
        return self.C.shape[1]

    @property
    def p(self) -> int:
        return self.V.shape[1]


def load_yelp_subset(dir_path: str | Path) -> TextData:
    """
    Load a dense Yelp subset produced by scripts/make_yelp_subset_for_idmr.py.

    Expected files in dir_path:
      - C.npy  (n, d) counts (dense)
      - V.npy  (n, p) covariates (dense)
      - M.npy  (n,)   total counts
    """
    dpath = Path(dir_path)
    C_path = dpath / "C.npy"
    V_path = dpath / "V.npy"
    M_path = dpath / "M.npy"

    if not C_path.exists():
        raise FileNotFoundError(f"Missing {C_path}")
    if not V_path.exists():
        raise FileNotFoundError(f"Missing {V_path}")
    if not M_path.exists():
        raise FileNotFoundError(f"Missing {M_path}")

    # The legacy engine expects float64 arrays.
    C = np.load(C_path).astype(np.float64, copy=False)
    V = np.load(V_path).astype(np.float64, copy=False)
    M = np.load(M_path).astype(np.float64, copy=False)

    return TextData.from_arrays(C=C, V=V, M=M)
