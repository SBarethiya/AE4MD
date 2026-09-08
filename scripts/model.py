from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MD-specific preprocessing
# ---------------------------------------------------------------------------

def kabsch_align(coords: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Align every frame in ``coords`` to a reference structure using the Kabsch algorithm.
    Parameters:
        coords : The coordinates of the ensemble to be aligned.
        reference : ndarray, shape (n_atoms, 3), optional The reference structure to align to. If None, the first frame of coords is used.
    Returns:
        aligned : The aligned coordinates.
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
    """
    Reshape (n_frames, n_atoms, 3) -> (n_frames, n_atoms * 3)
    """
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    return coords.reshape(n_frames, -1)


def unflatten_ensemble(flat: np.ndarray, n_atoms: int) -> np.ndarray:
    """
    Reshape (n_frames, n_atoms * 3) -> (n_frames, n_atoms, 3)
    """
    flat = np.asarray(flat)
    n_frames = flat.shape[0]
    n_features = flat.shape[1] // n_atoms
    return flat.reshape(n_frames, n_atoms, n_features)


class Standardizer:
    """
    Standardize features to zero mean and unit variance.
    """

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
    """
    Scale features to a given range (default [0, 1]) using min-max normalization.
    """
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
def resolve_layer_sizes(latent_dim: int, original_dim: int, hidden_dims: Optional[List[int]] = None,) -> List[int]:
    """
    Determine the sizes of the layers in the encoder/decoder MLPs.
    """
    return [latent_dim, *hidden_dims, original_dim]


def _mlp(sizes: List[int], activation: nn.Module = nn.ReLU(), final_activation: nn.Module = None) -> nn.Sequential:
    """
    Build a simple feedforward MLP with ReLU activations between layers.
    Parameters:
        sizes : Sizes of each layer, including input and output.
        final_activation : Optional activation function for the output layer.
    Returns:
        nn.Sequential model representing the MLP.
    """
    layers = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation)
    layers = layers[:-1]  # drop ReLU after the last linear layer
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    Maps input (original_dim) to a latent code (latent_dim)
    """

    def __init__(self, original_dim: int, latent_dim: int = 2, hidden_dims: Optional[List[int]] = None, 
                 activation: nn.Module = nn.ReLU()):
        super().__init__()
        sizes = resolve_layer_sizes(latent_dim, original_dim, hidden_dims)
        self.sizes = sizes
        self.net = _mlp(list(reversed(sizes)), activation=activation)  # original_dim -> ... -> latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Maps latent code (latent_dim) back to reconstruction (original_dim)
    """

    def __init__(self, original_dim: int, latent_dim: int = 2, hidden_dims: Optional[List[int]] = None, 
                 final_activation: Optional[nn.Module] = None, activation: nn.Module = nn.ReLU()):
        super().__init__()
        sizes = resolve_layer_sizes(latent_dim, original_dim, hidden_dims)
        self.sizes = sizes
        self.net = _mlp(sizes, final_activation=final_activation, activation=activation)  # latent_dim -> ... -> original_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    """
    A simple feedforward autoencoder with an encoder and decoder.
    """

    def __init__(self, original_dim: int, latent_dim: int = 2, hidden_dims: Optional[List[int]] = None,
                 activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.encoder = Encoder(original_dim, latent_dim, hidden_dims, activation=activation)
        self.decoder = Decoder(original_dim, latent_dim, hidden_dims, activation=activation)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def build_autoencoder(original_dim: int, latent_dim: int = 2, hidden_dims: Optional[List[int]] = None,
                      learning_rate: float = 1e-4, device: str = "cpu", lr_step_size: Optional[int] = None, lr_gamma: Optional[float] = None,
                      activation: nn.Module = nn.ReLU()) -> Tuple[Autoencoder, torch.optim.Optimizer]:
    """
    Construct an Autoencoder and its Adam optimizer, moved to device
    """
    model = Autoencoder(original_dim, latent_dim, hidden_dims, activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if lr_step_size is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma, )
    return model, optimizer, scheduler


def train_autoencoder(model: Autoencoder, optimizer: torch.optim.Optimizer, x_train: torch.Tensor, x_test: torch.Tensor,
                      epochs: int = 200, batch_size: int = 64, device: str = "cpu", patience: int = 20, min_delta: float = 1e-5,
                      scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None,) -> Tuple[List[float], List[float]]:
    """
    Train the autoencoder on the training data and evaluate on the test data.
    Parameters:
        model : The autoencoder model to train.
        optimizer : The optimizer for training.
        x_train : Training data tensor.
        x_test : Test data tensor.
        epochs : Number of training epochs.
        batch_size : Size of each training batch.
        device : Device to run the training on ('cpu' or 'cuda').
        patience : Number of epochs with no improvement after which training will be stopped.
        min_delta : Minimum change in the monitored quantity to qualify as an improvement.
    Returns:
        history : List of training losses per epoch.
        history_test : List of test losses per epoch.
    """

    model.train()
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(x_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_set = torch.utils.data.TensorDataset(x_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    history = []
    history_test = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        test_losses = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, x_hat = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                _, x_hat_test = model(batch)
                test_loss = criterion(x_hat_test, batch).item()
                test_losses += test_loss * batch.size(0)

        epoch_loss /= len(dataset)
        test_losses /= len(test_set)

        history.append(epoch_loss)
        history_test.append(test_losses)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}, val_loss: {history_test[-1]:.6f}")

        # Add early stopping logic
        if test_losses < best_val_loss - min_delta:
            best_val_loss = test_losses
            epochs_no_improve = 0
            best_state = model.state_dict()  # Save the best model state
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if scheduler is not None:
            scheduler.step()

        # Load the best model state before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, history_test

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_latent_space(z: np.ndarray, color: Optional[np.ndarray] = None, color_label: str = "frame index", 
                      title: str = "2D latent space of conformational ensemble", save_path: Optional[str] = None,):
    """
    Plot the 2D latent space representation of the ensemble.
    Parameters:
        z : ndarray, shape (n_samples, 2) The 2D latent representations of the samples.
        color : Optional ndarray, shape (n_samples,) Color values for each sample. If None, samples are colored by index.
        color_label : Label for the colorbar.
        title : Title of the plot.
        save_path : Optional path to save the figure. If None, the figure is not saved.
    Returns:
        fig, ax : The matplotlib figure and axes objects.
    """
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

def plot_losses(history: List[float], history_test: List[float], title: str = "Training and validation loss", save_path: Optional[str] = None,):
    """
    Plot the training and validation loss curves over epochs.
    Parameters:
        history : List of training losses per epoch.
        history_test : List of validation losses per epoch.
        title : Title of the plot.
        save_path : Optional path to save the figure. If None, the figure is not saved.
    Returns:
        fig, ax : The matplotlib figure and axes objects.
    """
    
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
    """
    Compute the root mean square deviation (RMSD) between a reference structure and a target structure.
    Parameters:
        reference : ndarray, shape (n_atoms, 3) or (n_frames, n_atoms, 3) The reference structure(s).
        target : ndarray, shape (n_frames, n_atoms, 3) The target structure(s).
    Returns:
        rmsd : ndarray, shape (n_frames,) The RMSD values for each frame in the target.
    """
    reference = np.asarray(reference)
    target = np.asarray(target)
    if reference.ndim == 2:
        reference = np.broadcast_to(reference, target.shape)
    sq_diff = np.sum((target - reference) ** 2, axis=-1)  # (n_frames, n_atoms)
    return np.sqrt(sq_diff.mean(axis=-1))  # (n_frames,)