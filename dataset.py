# dataset.py
import torch
import numpy as np

def build_dataset(u):
    nx, ny, nz = u.shape

    # Create grid points
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    pts = np.vstack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ]).T

    vals = u.ravel()[:, None]

    pts = torch.tensor(pts, dtype=torch.float32)
    vals = torch.tensor(vals, dtype=torch.float32)

    return pts, vals
