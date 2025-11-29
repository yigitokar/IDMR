from __future__ import annotations

from dataclasses import dataclass
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
