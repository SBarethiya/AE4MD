"""PyTorch autoencoder (AE) for 2D visualization of MD conformational ensembles.

Expected input: an ensemble of conformations from an MD trajectory, shape
``(n_frames, n_atoms, 3)``.

Pipeline
--------
1. ``kabsch_align``      - superpose every frame onto a reference structure to
                            remove rigid-body translation/rotation, so the AE
                            learns internal conformational variance instead of
                            whole-molecule drift.
2. ``flatten_ensemble``   - reshape (n_frames, n_atoms, 3) -> (n_frames, n_atoms*3).
3. ``Standardizer``       - zero-mean/unit-variance scaling, fit on train only.
4. ``Autoencoder``        - configurable number of layers/nodes (see previous
                            version), latent_dim=2 for direct visualization.
5. ``plot_latent_space``  - scatter plot of the 2D embedding.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# MD-specific preprocessing
# ---------------------------------------------------------------------------

def kabsch_align(coords: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Superpose every frame in ``coords`` onto ``reference`` (Kabsch algorithm),
    removing translational and rotational differences between frames.

    This matters for MD ensembles: raw Cartesian coordinates mix real
    conformational change with arbitrary whole-molecule translation/rotation.
    Aligning first means the AE's latent space reflects actual conformation.

    Parameters
    ----------
    coords : ndarray, shape (n_frames, n_atoms, 3)
    reference : ndarray, shape (n_atoms, 3), optional
        Structure to align to. Defaults to the first frame of ``coords``.

    Returns
    -------
    ndarray, shape (n_frames, n_atoms, 3)
        Aligned coordinates (each frame centered at its own centroid, then
        optimally rotated onto the reference).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if reference is None:
        reference = coords[0]
    ref_centered = reference - reference.mean(axis=0)

    aligned = np.empty_like(coords)
    for i, frame in enumerate(coords):
        frame_centered = frame - frame.mean(axis=0)
        H = frame_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        aligned[i] = (R @ frame_centered.T).T
    return aligned


def flatten_ensemble(coords: np.ndarray) -> np.ndarray:
    """Reshape ``(n_frames, n_atoms, 3)`` -> ``(n_frames, n_atoms * 3)``."""
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    return coords.reshape(n_frames, -1)


def unflatten_ensemble(flat: np.ndarray, n_atoms: int) -> np.ndarray:
    """Reshape ``(n_frames, n_atoms * 3)`` -> ``(n_frames, n_atoms, 3)``."""
    flat = np.asarray(flat)
    n_frames = flat.shape[0]
    return flat.reshape(n_frames, n_atoms, 3)


class Standardizer:
    """Zero-mean/unit-variance scaling. Fit on training data only, then reuse
    the same statistics for validation/test data and for un-scaling
    reconstructions/decoded samples back to real coordinate units."""

    def __init__(self):
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "Standardizer":
        self.mean_ = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        self.std_ = torch.where(std < 1e-8, torch.ones_like(std), std)  # avoid /0 on frozen atoms
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_ + self.mean_
    
class MinMaxScaler:
    def __init__(self):
        self.min_: Optional[torch.Tensor] = None
        self.max_: Optional[torch.Tensor] = None
    
    def fit(self, x: torch.Tensor) -> "MinMaxScaler":
        self.min_ = x.min(dim=0, keepdim=True)[0]
        self.max_ = x.max(dim=0, keepdim=True)[0]
        return self
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_) / (self.max_ - self.min_)
    
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit(x).transform(x)
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.max_ - self.min_) + self.min_


# ---------------------------------------------------------------------------
# Architecture (hyperparameter-driven, from the previous step)
# ---------------------------------------------------------------------------

def linear_layer_sizes(latent_dim: int, original_dim: int, num_hidden_layers: int) -> List[int]:
    """Evenly space ``num_hidden_layers`` layer sizes between ``latent_dim`` and
    ``original_dim`` (simple linear interpolation, no rounding tricks).

    Returns
    -------
    list[int]
        Layer sizes, ordered ``[latent_dim, ..., original_dim]``.
    """
    num_steps = num_hidden_layers + 1
    sizes = [
        round(latent_dim + (original_dim - latent_dim) * i / num_steps)
        for i in range(num_steps)
    ]
    sizes.append(original_dim)
    return sizes


def resolve_layer_sizes(
    latent_dim: int,
    original_dim: int,
    num_hidden_layers: int = 4,
    hidden_dims: Optional[List[int]] = None,
) -> List[int]:
    """Resolve the full list of layer sizes ``[latent_dim, ..., original_dim]``.

    - If ``hidden_dims`` is given, it's used directly (full control over the
      number of layers and nodes per layer).
    - Otherwise, ``num_hidden_layers`` evenly-spaced layers are generated via
      ``linear_layer_sizes``.
    """
    if hidden_dims is not None:
        return [latent_dim, *hidden_dims, original_dim]
    return linear_layer_sizes(latent_dim, original_dim, num_hidden_layers)


def _mlp(sizes: List[int], final_activation: nn.Module = None) -> nn.Sequential:
    """Build a stack of Linear + ReLU layers from consecutive ``sizes``, with an
    optional activation after the final layer."""
    layers = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    layers = layers[:-1]  # drop ReLU after the last linear layer
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Maps input (original_dim) to a latent code (latent_dim)."""

    def __init__(
        self,
        original_dim: int,
        latent_dim: int = 2,
        num_hidden_layers: int = 4,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        sizes = resolve_layer_sizes(latent_dim, original_dim, num_hidden_layers, hidden_dims)
        self.sizes = sizes
        self.net = _mlp(list(reversed(sizes)))  # original_dim -> ... -> latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Maps a latent code (latent_dim) back to the original space (original_dim).

    No final activation by default: MD coordinates are standardized
    (zero-mean/unit-variance), not bounded to [0, 1], so a Sigmoid output
    would be wrong here. Pass ``final_activation`` explicitly if you scale
    your data differently (e.g. min-max to [0, 1]).
    """

    def __init__(
        self,
        original_dim: int,
        latent_dim: int = 2,
        num_hidden_layers: int = 4,
        hidden_dims: Optional[List[int]] = None,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        sizes = resolve_layer_sizes(latent_dim, original_dim, num_hidden_layers, hidden_dims)
        self.sizes = sizes
        self.net = _mlp(sizes, final_activation=final_activation)  # latent_dim -> ... -> original_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    """Plain (non-variational) autoencoder combining an Encoder and Decoder.

    Architecture hyperparameters
    ----------------------------
    latent_dim : int
        Size of the bottleneck / latent code. Use 2 for direct 2D
        visualization of a conformational ensemble.
    num_hidden_layers : int
        Number of hidden layers to auto-generate (ignored if ``hidden_dims``
        is given).
    hidden_dims : list[int], optional
        Explicit number of nodes for each hidden layer. Its length sets the
        number of hidden layers; each entry sets the width of that layer.
    """

    def __init__(
        self,
        original_dim: int,
        latent_dim: int = 2,
        num_hidden_layers: int = 4,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.encoder = Encoder(original_dim, latent_dim, num_hidden_layers, hidden_dims)
        self.decoder = Decoder(original_dim, latent_dim, num_hidden_layers, hidden_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent_code, reconstruction)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def build_autoencoder(
    original_dim: int,
    latent_dim: int = 2,
    num_hidden_layers: int = 4,
    hidden_dims: Optional[List[int]] = None,
    learning_rate: float = 1e-4,
    device: str = "cpu",
) -> Tuple[Autoencoder, torch.optim.Optimizer]:
    """Construct an Autoencoder and its Adam optimizer, moved to ``device``."""
    model = Autoencoder(original_dim, latent_dim, num_hidden_layers, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def train_autoencoder(
    model: Autoencoder,
    optimizer: torch.optim.Optimizer,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[List[float], List[float]]:
    """Train ``model`` with MSE reconstruction loss.

    Parameters
    ----------
    x_train : torch.Tensor, shape (n_samples, original_dim)
        Flattened, aligned, standardized coordinates.

    Returns
    -------
    list[float], list[float]
        Mean training loss per epoch, mean validation loss per epoch.
    """
    model.train()
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(x_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    history_test = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, x_hat = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        with torch.no_grad():
            x_test = x_test.to(device)
            _, x_hat_test = model(x_test)
            test_loss = criterion(x_hat_test, x_test).item()
            test_loss *= x_test.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        history_test.append(test_loss / len(x_test))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}, val_loss: {history_test[-1]:.6f}")

    return history, history_test

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_latent_space(
    z: np.ndarray,
    color: Optional[np.ndarray] = None,
    color_label: str = "frame index",
    title: str = "2D latent space of conformational ensemble",
    save_path: Optional[str] = None,
):
    """Scatter-plot a 2D latent embedding of the ensemble.

    Parameters
    ----------
    z : ndarray, shape (n_frames, 2)
    color : ndarray, shape (n_frames,), optional
        Values to color points by (e.g. frame index / time, RMSD to a
        reference, a collective variable, cluster label). Defaults to frame
        index if not given.
    """
    import matplotlib.pyplot as plt

    z = np.asarray(z)
    if color is None:
        color = np.arange(z.shape[0])

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=color, cmap="viridis", s=12, alpha=0.8)
    fig.colorbar(sc, ax=ax, label=color_label)
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig, ax

def plot_losses(
    history: List[float],
    history_test: List[float],
    title: str = "Training and validation loss",
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history, label="train")
    ax.plot(history_test, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig, ax

def RMSD(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame RMSD between ``target`` frames and ``reference``, in whatever
    length units the input coordinates use (e.g. Angstrom).
 
    Parameters
    ----------
    reference : ndarray, shape (n_atoms, 3) or (n_frames, n_atoms, 3)
        A single reference structure (broadcast to every frame), or a
        per-frame reference — e.g. the original aligned ensemble, to score
        how well an autoencoder's reconstructions match frame-by-frame.
    target : ndarray, shape (n_frames, n_atoms, 3)
 
    Returns
    -------
    ndarray, shape (n_frames,)
        RMSD of each frame in ``target`` to its corresponding reference.
    """
    reference = np.asarray(reference)
    target = np.asarray(target)
    if reference.ndim == 2:
        reference = np.broadcast_to(reference, target.shape)
    sq_diff = np.sum((target - reference) ** 2, axis=-1)  # (n_frames, n_atoms)
    return np.sqrt(sq_diff.mean(axis=-1))  # (n_frames,)

# ---------------------------------------------------------------------------
# Example end-to-end pipeline (synthetic data standing in for an MD ensemble)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # --- stand-in for a real trajectory: (n_frames, n_atoms, 3) ---
    n_frames, n_atoms = 2000, 50
    t = np.linspace(0, 4 * np.pi, n_frames)
    base = np.random.randn(n_atoms, 3) * 3  # a fixed-ish reference structure
    coords = np.stack(
        [base + 0.3 * np.sin(t[i]) * np.random.randn(n_atoms, 3) for i in range(n_frames)]
    )
    # inject arbitrary rigid-body translation/rotation per frame, as in real MD
    for i in range(n_frames):
        shift = np.random.randn(3) * 5
        coords[i] += shift

    # --- 1. remove rigid-body motion ---
    aligned = kabsch_align(coords, reference=coords[0])

    # --- 2. flatten ---
    flat = flatten_ensemble(aligned)  # (n_frames, n_atoms*3)

    # --- 3. train/test split + standardize (fit on train only) ---
    n_train = int(0.8 * n_frames)
    idx = np.random.permutation(n_frames)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    x_train = torch.tensor(flat[train_idx], dtype=torch.float32)
    x_test = torch.tensor(flat[test_idx], dtype=torch.float32)

    if scaler_type == "standard":
        scaler = Standardizer()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        x_train_scaled = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
        x_test_scaled = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    else:
        x_train_scaled = x_train
        x_test_scaled = x_test

    # --- 4. build + train AE, latent_dim=2 for visualization ---
    original_dim = flat.shape[1]  # n_atoms * 3
    model, optimizer = build_autoencoder(
        original_dim=original_dim,
        latent_dim=2,
        hidden_dims=[32, 64, 128],  # or use num_hidden_layers=4 for auto sizing
        learning_rate=1e-3,
    )
    train_autoencoder(model, optimizer, x_train_scaled, x_test_scaled, epochs=50, batch_size=64)

    # --- 5. encode the full ensemble and plot ---
    model.eval()
    with torch.no_grad():
        full_scaled = scaler.transform(torch.tensor(flat, dtype=torch.float32))
        z, _ = model(full_scaled)
    z = z.numpy()

    RMSD(coords[0], unflatten_ensemble(scaler.inverse_transform(model.decoder(z)).numpy(), n_atoms))

    plot_latent_space(z, color=np.arange(n_frames), color_label="frame index",
                       save_path="latent_space.png")
    print("Saved latent_space.png")
    plot_losses(history, history_test, save_path="loss_curves.png")
    print("Saved loss_curves.png")
    