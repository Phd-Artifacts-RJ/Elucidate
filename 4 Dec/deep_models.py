# deep_models.py
"""
PyTorch-based classifiers that are scikit-learn compatible.

All estimators here assume that preprocessing (imputation, scaling,
encoding, etc.) is done upstream in a scikit-learn Pipeline.

For temporal models (TCN / TransformerEncoder), we interpret X
as (n_samples, n_features) but reshape it into sequences:

    n_features = seq_len * feature_dim

so we require n_features % seq_len == 0.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = TensorDataset = None


# -------------------------------------------------------------------
# Shared base utilities
# -------------------------------------------------------------------

class _TorchBaseClassifier(BaseEstimator, ClassifierMixin):
    """
    Base mixin for torch-based classifiers.

    Provides:
    - torch presence check
    - device selection with a runtime print
    - numpy -> tensor conversion
    """

    def __init__(self, verbose=True, device=None, random_state=42):
        self.verbose = verbose
        self.device = device
        self.random_state = random_state
        self._fitted = False
        self._device_used = None
        self.classes_ = np.array([0, 1], dtype=int)

    # ---------- core utilities ----------

    def _check_torch(self):
        if not TORCH_AVAILABLE:
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch. "
                f"Install `torch` to use this model."
            )

    def _prepare_device(self):
        if self.device is not None:
            device = self.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"\n[ {self.__class__.__name__} ] Using device: {device.upper()}")
        self._device_used = device
        return device

    def _set_seeds(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _to_tensor_2d(self, X, y=None):
        """X: array-like (n_samples, n_features)."""
        X_arr = np.asarray(X)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        if y is None:
            return X_tensor
        y_arr = np.asarray(y).reshape(-1, 1)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)
        return X_tensor, y_tensor

    def _to_tensor_3d_seq(self, X, seq_len, y=None):
        """
        Interpret X (n_samples, n_features) as a sequence:
        n_features = seq_len * feature_dim, reshape to
            (n_samples, seq_len, feature_dim).
        """
        X_arr = np.asarray(X)
        n_samples, n_features = X_arr.shape
        if n_features % seq_len != 0:
            raise ValueError(
                f"n_features={n_features} is not divisible by seq_len={seq_len}. "
                f"Provide compatible features or adjust seq_len."
            )
        d_feat = n_features // seq_len
        X_seq = X_arr.reshape(n_samples, seq_len, d_feat)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        if y is None:
            return X_tensor

        y_arr = np.asarray(y).reshape(-1, 1)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)
        return X_tensor, y_tensor


# -------------------------------------------------------------------
# 1. TorchMLPClassifier – simple feedforward MLP (tabular)
# -------------------------------------------------------------------

class TorchMLPClassifier(_TorchBaseClassifier):
    """
    Simple fully-connected network for binary classification
    on tabular data (no temporal structure assumed).

    X is treated as (n_samples, n_features) directly.
    """

    def __init__(
        self,
        hidden_dims=(64, 32),
        dropout=0.1,
        lr=1e-3,
        batch_size=256,
        max_epochs=50,
        patience=5,
        verbose=True,
        device=None,
        random_state=42,
    ):
        super().__init__(verbose=verbose, device=device, random_state=random_state)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        self._model = None
        self._input_dim = None

    def _build_model(self, input_dim):
        layers = []
        in_dim = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # single logit
        return nn.Sequential(*layers)

    def fit(self, X, y):
        self._check_torch()
        self._set_seeds()
        device = self._prepare_device()

        X_tensor, y_tensor = self._to_tensor_2d(X, y)
        n_samples, n_features = X_tensor.shape

        self._input_dim = n_features
        self._model = self._build_model(n_features).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            running_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / n_samples

            if self.verbose:
                print(
                    f"[TorchMLPClassifier] Epoch {epoch+1:03d}/{self.max_epochs} "
                    f"| Loss: {epoch_loss:.5f}"
                )

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.patience and epochs_no_improve >= self.patience:
                    if self.verbose:
                        print("[TorchMLPClassifier] Early stopping.")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.to(device)
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("TorchMLPClassifier is not fitted yet.")

        device = self._device_used or self._prepare_device()
        X_tensor = self._to_tensor_2d(X).to(device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_tensor)
            probs_pos = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# -------------------------------------------------------------------
# 2. TorchTCNClassifier – temporal convolutional network (TCN)
#     (captures local + medium-range temporal dependencies)
# -------------------------------------------------------------------

class _TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        # causal: trim to original length
        out = out[..., : x.size(-1)]
        return out + res


class TorchTCNClassifier(_TorchBaseClassifier):
    """
    TCN-based classifier.

    X is (n_samples, n_features) and is reshaped internally to
    (n_samples, seq_len, feature_dim).

    - seq_len controls how many time steps you assume.
    - feature_dim = n_features / seq_len (must be integer).

    This is conceptually aligned with the TCN component in the
    TCN-DilateFormer hybrid model for credit risk. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        seq_len=11,
        channels=(64, 64, 64),
        kernel_size=3,
        dropout=0.1,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        patience=5,
        verbose=True,
        device=None,
        random_state=42,
    ):
        super().__init__(verbose=verbose, device=device, random_state=random_state)
        self.seq_len = seq_len
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        self._model = None
        self._feature_dim = None

    def _build_tcn(self, in_channels):
        layers = []
        prev_c = in_channels
        for i, c in enumerate(self.channels):
            dilation = 2 ** i
            layers.append(
                _TemporalBlock(
                    in_channels=prev_c,
                    out_channels=c,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                )
            )
            prev_c = c
        tcn = nn.Sequential(*layers)
        head = nn.Linear(prev_c, 1)
        return tcn, head

    def fit(self, X, y):
        self._check_torch()
        self._set_seeds()
        device = self._prepare_device()

        X_tensor, y_tensor = self._to_tensor_3d_seq(X, self.seq_len, y)
        # X_tensor: (B, T, F) -> for Conv1d we need (B, C, T) with C=F
        B, T, F = X_tensor.shape
        self._feature_dim = F
        X_tensor = X_tensor.permute(0, 2, 1)  # (B, F, T)

        tcn, head = self._build_tcn(in_channels=F)
        self._tcn = tcn.to(device)
        self._head = head.to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        params = list(self._tcn.parameters()) + list(self._head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self._tcn.train()
            self._head.train()
            running_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                feats = self._tcn(xb)          # (B, C_last, T)
                feats = feats.mean(dim=2)      # global average over time -> (B, C_last)
                logits = self._head(feats)     # (B,1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / B

            if self.verbose:
                print(
                    f"[TorchTCNClassifier] Epoch {epoch+1:03d}/{self.max_epochs} "
                    f"| Loss: {epoch_loss:.5f}"
                )

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = {
                    "tcn": {k: v.cpu().clone() for k, v in self._tcn.state_dict().items()},
                    "head": {k: v.cpu().clone() for k, v in self._head.state_dict().items()},
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.patience and epochs_no_improve >= self.patience:
                    if self.verbose:
                        print("[TorchTCNClassifier] Early stopping.")
                    break

        if best_state is not None:
            self._tcn.load_state_dict(best_state["tcn"])
            self._head.load_state_dict(best_state["head"])
        self._tcn.to(device)
        self._head.to(device)
        self._fitted = True
        return self

    def _forward_logits(self, X):
        device = self._device_used or self._prepare_device()
        X_tensor = self._to_tensor_3d_seq(X, self.seq_len)  # (B,T,F)
        X_tensor = X_tensor.permute(0, 2, 1)  # (B,F,T)
        X_tensor = X_tensor.to(device)

        self._tcn.eval()
        self._head.eval()
        with torch.no_grad():
            feats = self._tcn(X_tensor)
            feats = feats.mean(dim=2)
            logits = self._head(feats)
        return logits

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("TorchTCNClassifier is not fitted yet.")
        logits = self._forward_logits(X)
        probs_pos = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# -------------------------------------------------------------------
# 3. TorchTransformerEncoderClassifier – encoder-only Transformer
#     (captures long-range temporal dependencies)
# -------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (T, B, d_model)
        """
        T = x.size(0)
        return x + self.pe[:T]


class TorchTransformerEncoderClassifier(_TorchBaseClassifier):
    """
    Transformer encoder for binary classification on sequences.

    X is (n_samples, n_features) reshaped into (n_samples, seq_len, feature_dim).

    This corresponds to the kind of architecture used in Hu & Yeo's
    transformer-based credit card default model. :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        seq_len=13,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        patience=5,
        verbose=True,
        device=None,
        random_state=42,
    ):
        super().__init__(verbose=verbose, device=device, random_state=random_state)
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        self._feature_dim = None
        self._input_proj = None
        self._encoder = None
        self._pos_encoder = None
        self._head = None

    def _build_model(self, feature_dim):
        self._feature_dim = feature_dim
        self._input_proj = nn.Linear(feature_dim, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=False,  # we will use (T,B,dim)
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )
        self._pos_encoder = _PositionalEncoding(self.d_model)
        self._head = nn.Linear(self.d_model, 1)

    def fit(self, X, y):
        self._check_torch()
        self._set_seeds()
        device = self._prepare_device()

        X_tensor, y_tensor = self._to_tensor_3d_seq(X, self.seq_len, y)
        # X_tensor: (B, T, F)
        B, T, F = X_tensor.shape

        self._build_model(feature_dim=F)
        self._input_proj.to(device)
        self._encoder.to(device)
        self._pos_encoder.to(device)
        self._head.to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        params = list(self._input_proj.parameters()) + \
                 list(self._encoder.parameters()) + \
                 list(self._head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self._input_proj.train()
            self._encoder.train()
            self._head.train()
            running_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)  # xb: (B,T,F)

                optimizer.zero_grad()
                # project + positional encoding
                x_proj = self._input_proj(xb)          # (B,T,d_model)
                x_proj = x_proj.transpose(0, 1)        # (T,B,d_model)
                x_pe = self._pos_encoder(x_proj)       # (T,B,d_model)

                enc_out = self._encoder(x_pe)          # (T,B,d_model)
                pooled = enc_out.mean(dim=0)           # (B,d_model)
                logits = self._head(pooled)            # (B,1)

                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / B

            if self.verbose:
                print(
                    f"[TorchTransformerEncoderClassifier] "
                    f"Epoch {epoch+1:03d}/{self.max_epochs} | Loss: {epoch_loss:.5f}"
                )

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = {
                    "input_proj": {k: v.cpu().clone() for k, v in self._input_proj.state_dict().items()},
                    "encoder": {k: v.cpu().clone() for k, v in self._encoder.state_dict().items()},
                    "head": {k: v.cpu().clone() for k, v in self._head.state_dict().items()},
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.patience and epochs_no_improve >= self.patience:
                    if self.verbose:
                        print("[TorchTransformerEncoderClassifier] Early stopping.")
                    break

        if best_state is not None:
            self._input_proj.load_state_dict(best_state["input_proj"])
            self._encoder.load_state_dict(best_state["encoder"])
            self._head.load_state_dict(best_state["head"])

        self._input_proj.to(device)
        self._encoder.to(device)
        self._head.to(device)
        self._fitted = True
        return self

    def _forward_logits(self, X):
        device = self._device_used or self._prepare_device()
        X_tensor = self._to_tensor_3d_seq(X, self.seq_len).to(device)  # (B,T,F)

        self._input_proj.eval()
        self._encoder.eval()
        self._head.eval()

        with torch.no_grad():
            x_proj = self._input_proj(X_tensor)   # (B,T,d_model)
            x_proj = x_proj.transpose(0, 1)      # (T,B,d_model)
            x_pe = self._pos_encoder(x_proj)
            enc_out = self._encoder(x_pe)
            pooled = enc_out.mean(dim=0)         # (B,d_model)
            logits = self._head(pooled)          # (B,1)
        return logits

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("TorchTransformerEncoderClassifier is not fitted yet.")
        logits = self._forward_logits(X)
        probs_pos = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
