import itertools, pickle
import argparse, ast
import numpy as np
import torch
from pathlib import Path

from scripts.model import (kabsch_align, flatten_ensemble, unflatten_ensemble, Standardizer, MinMaxScaler, build_autoencoder, train_autoencoder, plot_latent_space, plot_losses, RMSD,)

OUTPUT_DIR = "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}")

parser = argparse.ArgumentParser(description="Input preparation of the model for protein dynamics", formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars="@")

parser.add_argument("--output_dir", help="output directory (format: dir)", required=True)
parser.add_argument("--save_file_name", help="file name to save the output (format: str)", required=True)
parser.add_argument("--BATCH_SIZE", help="Batch size list (format: list)", required=False, type=parse_list, default=[32, 64])
parser.add_argument("--train_test_split", help="Train/test split ratio (format: float)", required=False, type=float, default=0.8)
parser.add_argument("--LATENT_DIM", help="Latent dimension list (format: list)", required=False, type=parse_list, default=[2, 4, 8])
parser.add_argument("--NUM_HIDDEN_LAYER", help="Number of hidden layers list (format: list)", required=False, type=parse_list, default=[2, 4])
parser.add_argument("--HIDDEN_DIMS", help="Hidden dimensions list (format: list)", required=False, type=parse_list, default=[[64, 128]])
parser.add_argument("--LEARNING_RATE", help="List of hidden-layer-width lists to try, e.g. '[[64,128],[32,64,128]]' (format: list of lists)", required=False, type=parse_list, default=None)
parser.add_argument("--EPOCHS", help="Number of epochs list (format: list)", required=False, type=parse_list, default=[100, 200])
parser.add_argument("--scalar_type", help="scaler type (standard, minmax, None)", required=False, default=None)
parser.add_argument("--seed", help="random seed (format: int)", required=False, default=42)
args = parser.parse_args()

# Hyperparameter grid: every combination of these values is trained and evaluated.
def gen_parms_combinations(**params_dict):
    """
    params_dit: searching grid for each hyperparameter
        it has to contain the following keys:
        BATCH_SIZE: [64,]
        LATENT_DIM: [2, 5, 10]
        NUM_HIDDEN_LAYER: [3, 4, 5]
        EPOCHS: [100, 200, 500]
        RATE: [0.001, 0.01, 0.0001]
    return hyperparams_combinations
        <-- list of hyperparameter dict with BATCH_SIZE LATENT_DIM NUM_HIDDEN_LAYER EPOCHS RATE as the keys
    """
    hyperparams_list = [
                  params_dict['BATCH_SIZE'],
                  params_dict['LATENT_DIM'],
                  params_dict['NUM_HIDDEN_LAYER'],
                  params_dict['HIDDEN_DIMS'],
                  params_dict['EPOCHS'],
                  params_dict['LEARNING_RATE'],
                 ]
    combinations = list(itertools.product(*hyperparams_list))

    # reconfigure combinations to dict
    hyperparams_combinations = []
    for i in combinations:
        dict_tmp = {}
        dict_tmp['BATCH_SIZE'], dict_tmp['LATENT_DIM'], dict_tmp['NUM_HIDDEN_LAYER'], dict_tmp["HIDDEN_DIMS"], dict_tmp['EPOCHS'], dict_tmp['LEARNING_RATE'] = i

        hyperparams_combinations.append(dict_tmp)
    return hyperparams_combinations

def load_dataset(output_dir, file_name, train_test_split):
    """Load the training and test datasets, plus the scaler used to normalize/standardize them."""
    print(f"Loading dataset from {output_dir}/{file_name} files...\n")
    coords_original =  np.array(np.load(f"{output_dir}/{file_name}.pkl", allow_pickle=True))
    print(f"Loaded {coords_original.shape[0]} frames of {coords_original.shape} atoms each.")

    # remove the name of the sequence that is the 4rth column in the coords array: shape(n_frames, n_atoms, 4) -> shape(n_frames, n_atoms, 3)
    coords = coords_original[:, :, :3]
    sequence_names = coords_original[:, :, 3]
    # sequece_names convert to frame-wise sequence names: shape(n_frames, n_atoms) -> shape(n_frames,)
    sequence_names = np.array([sequence_names[i, 0] for i in range(sequence_names.shape[0])])

    # --- 1. remove rigid-body motion ---
    aligned = kabsch_align(coords, reference=coords[0])

    # --- 2. flatten ---
    flat = flatten_ensemble(aligned)  # (n_frames, n_atoms*3)
    
    # --- 3. train/test split + standardize (fit on train only) ---

    n_frames = coords.shape[0]
    n_train = int(train_test_split * n_frames)
    idx = np.random.permutation(n_frames)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    x_train = torch.tensor(flat[train_idx], dtype=torch.float32)
    x_test = torch.tensor(flat[test_idx], dtype=torch.float32)

    return x_train, x_test, coords, sequence_names

def run_single_config(config, x_train, x_test, raw_data, sequence_names, scaler, run_id, output_dir):
    """Train one AE with ``config``, plot diagnostics, and return evaluation metrics."""
    original_dim = x_train.shape[1]
    n_frames = x_train.shape[0] + x_test.shape[0]
    flat = torch.tensor(flatten_ensemble(raw_data), dtype=torch.float32).to(DEVICE)

    if config["HIDDEN_DIMS"] is None:
        model, optimizer = build_autoencoder(
        original_dim=original_dim,
        latent_dim=config["LATENT_DIM"],
        num_hidden_layers=config["NUM_HIDDEN_LAYER"],
        learning_rate=config["LEARNING_RATE"],
        device=DEVICE,
    )
    else:
        model, optimizer = build_autoencoder(
            original_dim=original_dim,
            latent_dim=config["LATENT_DIM"],
            hidden_dims=config["HIDDEN_DIMS"],
            num_hidden_layers=config["NUM_HIDDEN_LAYER"],
            learning_rate=config["LEARNING_RATE"],
            device=DEVICE,
        )

    history, history_test = train_autoencoder(
        model,
        optimizer,
        x_train,
        x_test,
        epochs=config["EPOCHS"],
        batch_size=config["BATCH_SIZE"],
        device=DEVICE,
    )

    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        z, reconstructed = model(flat)
    z_latent = z.cpu().numpy()

    plot_latent_space(z_latent, color=sequence_names, color_label="sequence_names",
                       save_path=f"{run_dir}/latent_space.png")
    print("Saved latent_space.png")
    plot_losses(history, history_test, save_path=f"{run_dir}/loss_curves.png")
    print("Saved loss_curves.png")

    n_atoms = int(original_dim / 3)
    # device the reconstructed data back to the original scale if a scaler was used
    
    if scaler is not None:
        reconstructed = reconstructed.cpu()
        reconstructed = scaler.inverse_transform(reconstructed)
        rmsd = RMSD(raw_data, unflatten_ensemble(reconstructed, n_atoms))
    else:
        reconstructed = reconstructed.cpu().numpy()
        rmsd = RMSD(raw_data, unflatten_ensemble(reconstructed, n_atoms))

    metrics = {"run_id": run_id, "final_train_loss": history[-1], "final_test_loss": history_test[-1], "RMSD": rmsd, "rmsd_mean": np.mean(rmsd), "rmsd_std": np.std(rmsd), **config}

    with open(f"{run_dir}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # save the model
    torch.save(model.state_dict(), f"{run_dir}/model.pth")

    return metrics


def grid_search(x_train, x_test, raw_data, sequence_names, scaler, hyperparams_dict, output_dir):
    """Train and evaluate every combination in ``PARAM_GRID``.

    Returns
    -------
    list[dict]
        One metrics dict per configuration, sorted by ``rmsd_mean`` ascending.
    """
    n_atoms = x_train.shape[-1] 
    hyperparams_combinations = gen_parms_combinations(**hyperparams_dict)
    print(f"Running grid search over {len(hyperparams_combinations)} configurations...")

    results = []
    for i, config in enumerate(hyperparams_combinations):
        run_id = f"run_{i + 1:03d}_ld{config['LATENT_DIM']}_layer{config['NUM_HIDDEN_LAYER']}_B{config['BATCH_SIZE']}_E{config['EPOCHS']}_lr{config['LEARNING_RATE']}"
        print(f"\n[{i + 1}/{len(config)}], config={config}")
 
        metrics = run_single_config(config, x_train, x_test, raw_data, sequence_names, scaler, run_id, output_dir)
        results.append(metrics)
        print(f"  -> train_loss={metrics['final_train_loss']:.4f}  rmsd_mean={metrics['rmsd_mean']:.4f}")

    results.sort(key=lambda m: m["rmsd_mean"])
    return results

hyperparams_dict = {
    "BATCH_SIZE": args.BATCH_SIZE,
    "LATENT_DIM": args.LATENT_DIM,
    "NUM_HIDDEN_LAYER": args.NUM_HIDDEN_LAYER,
    "EPOCHS": args.EPOCHS,
    "LEARNING_RATE": args.LEARNING_RATE,
    "HIDDEN_DIMS": args.HIDDEN_DIMS,
}

x_train, x_test, raw_data, sequence_names = load_dataset(args.output_dir, args.save_file_name, float(args.train_test_split))

if args.scalar_type == "standard":
    scaler = Standardizer()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
elif args.scalar_type == "minmax":
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
else:
    scaler = None
    x_train_scaled = x_train
    x_test_scaled = x_test

results = grid_search(x_train_scaled, x_test_scaled, raw_data, sequence_names, scaler, hyperparams_dict, args.output_dir)

summary_path = f"{args.output_dir}/grid_search_summary.pkl"

with open(summary_path, "wb") as f:
    pickle.dump(results, f)

best = results[0]
print("\nBest configuration (lowest RMSD):")
print(best)