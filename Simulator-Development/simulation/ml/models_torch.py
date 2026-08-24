"""
PyTorch models for ground-effect factor prediction, plus a generic training
loop shared by all of them. Two input shapes are used:

  - pointwise:  [N, F]     -- a single timestep's sensor reading (MLPNet)
  - sequence:   [N, W, F]  -- the last W timesteps of sensor history
                              (LSTMNet, TransformerNet)

The sequence models exist because this data is a real time series (a
continuous flight): a single instantaneous reading can't distinguish "flying
over a slope" from "flying over a bump" as well as the recent trend in
motor current/gyro can.
"""
import numpy as np
import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):  # x: [B, F]
        return self.net(x).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):  # x: [B, W, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # representation at the most recent timestep
        return self.head(last).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, n_features: int, window: int, d_model: int = 32, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):  # x: [B, W, F]
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


def fit_torch_model(model, X_train, y_train, X_val, y_val, epochs=25, lr=1e-3,
                     batch_size=256, patience=6, device="cpu", verbose=True):
    """Generic training loop: Adam + MSE, early stopping on a validation
    split carved out of *training* data (never the held-out test set --
    peeking at test performance to pick the best epoch would be leakage).
    Restores the best-validation-RMSE weights before returning."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_rmse = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        # Batched like predict_torch_model, not one giant forward pass over
        # the whole validation set -- for sequence models (LSTM/Transformer)
        # a single-batch forward over tens of thousands of windows can try
        # to allocate a multi-GB intermediate tensor and OOM on a GPU with
        # limited VRAM, even though the same call is merely slow on CPU.
        val_pred = predict_torch_model(model, X_val, device=device)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))

        if val_rmse < best_rmse - 1e-7:
            best_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            print(f"    epoch {epoch + 1:3d}/{epochs}  val_rmse={val_rmse:.5f}"
                  f"{'  *' if bad_epochs == 0 else ''}")
        if bad_epochs >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch + 1} (best val_rmse={best_rmse:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_torch_model(model, X, device="cpu", batch_size=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros((0,), dtype=np.float32)
