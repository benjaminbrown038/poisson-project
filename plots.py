# plots.py
import matplotlib.pyplot as plt
import numpy as np

def plot_slice(u, outdir, axis="z"):
    if axis == "z":
        mid = u.shape[2] // 2
        slice_ = u[:, :, mid]
    elif axis == "y":
        mid = u.shape[1] // 2
        slice_ = u[:, mid, :]
    else:
        mid = u.shape[0] // 2
        slice_ = u[mid, :, :]

    plt.figure(figsize=(6,5))
    plt.imshow(slice_, origin="lower", extent=[-1,1,-1,1])
    plt.colorbar(label="u(x,y,z)")
    plt.title(f"Slice through {axis}={mid}")
    plt.savefig(f"{outdir}/slice.png", dpi=200)
    plt.close()
