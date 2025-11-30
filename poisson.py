# poisson_physics.py
import numpy as np

def solve_poisson(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    tol, max_iters = cfg["tol"], cfg["max_iters"]

    # Grid
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Forcing term (Gaussian source)
    f = np.exp(-(X**2 + Y**2 + Z**2) * 8)

    # Initialize u
    u = np.zeros((nx, ny, nz))

    # Jacobi iteration
    for _ in range(max_iters):
        u_old = u.copy()

        u[1:-1, 1:-1, 1:-1] = (
            u_old[:-2, 1:-1, 1:-1] +
            u_old[2:, 1:-1, 1:-1] +
            u_old[1:-1, :-2, 1:-1] +
            u_old[1:-1, 2:, 1:-1] +
            u_old[1:-1, 1:-1, :-2] +
            u_old[1:-1, 1:-1, 2:] -
            f[1:-1, 1:-1, 1:-1]
        ) / 6.0

        if np.linalg.norm(u - u_old) < tol:
            break

    return u
