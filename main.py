# main_poisson.py
import os
from poisson_physics import solve_poisson
from dataset import build_dataset
from models import MLP
from train import train_model
from plots import plot_slice

def main():
    cfg = {
        "nx": 40, "ny": 40, "nz": 40,
        "tol": 1e-6, "max_iters": 3000,
        "hidden": 256, "depth": 4,
        "lr": 8e-4, "epochs": 800,
        "outdir": "experiments/results/poisson"
    }

    os.makedirs(cfg["outdir"], exist_ok=True)

    # --- Physics ground truth ---
    u = solve_poisson(cfg)

    # --- Dataset ---
    pts, vals = build_dataset(u)

    # --- Surrogate ---
    model = MLP(cfg["hidden"], cfg["depth"])
    train_model(model, pts, vals, cfg)

    # --- Plotting ---
    plot_slice(u, cfg["outdir"])

if __name__ == "__main__":
    main()
