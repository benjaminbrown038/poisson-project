# train.py
import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, pts, vals, cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pts = pts.to(device)
    vals = vals.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        pred = model(pts)
        loss = loss_fn(pred, vals)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss = {loss.item():.6f}")

    return model
