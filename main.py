import itertools, pickle
import argparse, ast
import numpy as np
import pandas as pd
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
parser.add_argument("--lr_step_size", help="Step size for learning rate scheduler (format: int)", required=False, type=int, default=100)
parser.add_argument("--lr_gamma", help="Gamma for learning rate scheduler (format: float)", required=False, type=float, default=0.5)
parser.add_argument("--activation", nargs="+", help="Activation function for the autoencoder (format: str)", required=False, default="relu")
parser.add_argument("--add_padding_indicator", help="Whether to add a padding indicator to the dataset (format: bool)", required=False, action="store_true")
args = parser.parse_args()

activations = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "gelu": torch.nn.GELU(),
}

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
        activation: ["relu", "sigmoid", "tanh", "gelu"]
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
                  params_dict['activation']
                 ]
    combinations = list(itertools.product(*hyperparams_list))

    # reconfigure combinations to dict
    hyperparams_combinations = []
    for i in combinations:
        dict_tmp = {}
        dict_tmp['BATCH_SIZE'], dict_tmp['LATENT_DIM'], dict_tmp['NUM_HIDDEN_LAYER'], dict_tmp["HIDDEN_DIMS"], dict_tmp['EPOCHS'], dict_tmp['LEARNING_RATE'], dict_tmp['activation'] = i

        hyperparams_combinations.append(dict_tmp)
    return hyperparams_combinations

def load_dataset(output_dir, file_name, train_test_split, add_padding_indicator, scaler):
    """
    Load the dataset from the specified output directory and file name, remove rigid-body motion, flatten the data, and split it into training and testing sets.
    Parameters:
        output_dir : The directory where the dataset file is located.
        file_name : The name of the dataset file (without extension).
        train_test_split : The ratio of training data to the total dataset (between 0 and 1).
    Returns:
        x_train : The training dataset as a PyTorch tensor.
        x_test : The testing dataset as a PyTorch tensor.
        coords : The original coordinates of the dataset after removing rigid-body motion.
        sequence_names : The sequence names corresponding to each frame in the dataset.
    """

    if add_padding_indicator:
        file_name = f"{file_name}_padded_with_indicator"
    print(f"Loading dataset from {output_dir}/{file_name} files...\n")
    coords_original =  np.array(np.load(f"{output_dir}/{file_name}.pkl", allow_pickle=True))
    print(f"Loaded {coords_original.shape[0]} frames of {coords_original.shape} atoms each.")

    # remove the name of the sequence that is the 4th column in the coords array: shape(n_frames, n_atoms, 4) -> shape(n_frames, n_atoms, 3)

    coords = coords_original[:, :, :3]
    if add_padding_indicator:
        sequence_names = coords_original[:, :, -1]
        padded_indices = coords_original[:, :, -2]
    else: 
        sequence_names = coords_original[:, :, -1]
        padded_indices = None

    # sequece_names convert to frame-wise sequence names: shape(n_frames, n_atoms) -> shape(n_frames,)
    sequence_names = np.array([sequence_names[i, 0] for i in range(sequence_names.shape[0])])

    # --- 1. remove rigid-body motion ---
    aligned = kabsch_align(coords, reference=coords[0])

    # ---2. check scaler type and scale the data ---
    if scaler == "standard":
        scaler = Standardizer()
        # aligned_flattened = flatten_ensemble(aligned)
        aligned_tensor = torch.as_tensor(aligned, dtype=torch.float32)
        aligned_scaled = scaler.fit_transform(aligned_tensor)
        # unflatten_aligned_scaled = unflatten_ensemble(aligned_scaled, aligned.shape[1])
    elif scaler == "minmax":
        scaler = MinMaxScaler()
        aligned_tensor = torch.as_tensor(aligned, dtype=torch.float32)
        # aligned_flattened = flatten_ensemble(aligned)
        aligned_scaled = scaler.fit_transform(aligned)
        # unflatten_aligned_scaled = unflatten_ensemble(aligned_scaled, aligned.shape[1])
    else:
        aligned_scaled = aligned
        scaler = None

    if add_padding_indicator:
        aligned_scaled = np.concatenate((aligned_scaled, padded_indices[:, :, np.newaxis]), axis=2)

    # --- 3. flatten ---
    flat = flatten_ensemble(aligned_scaled)  # (n_frames, n_atoms*3) or (n_frames, n_atoms*4) if add_padding_indicator is True
    
    # --- 4. train/test split + standardize (fit on train only) ---
    n_frames = coords.shape[0]
    n_train = int(train_test_split * n_frames)
    idx = np.random.permutation(n_frames)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    x_train = torch.tensor(flat[train_idx], dtype=torch.float32)
    x_test = torch.tensor(flat[test_idx], dtype=torch.float32)

    # Keep sequence names consistent with x_train and x_test 
    # sequence_names_train = sequence_names[train_idx] 
    # sequence_names_test = sequence_names[test_idx]

    return x_train, x_test, aligned_scaled, sequence_names, scaler

def run_single_config(config, x_train, x_test, raw_data, sequence_names, scaler, run_id, output_dir, add_padding_indicator, lr_step_size=100, lr_gamma=0.5):
    """
    Train and evaluate a single configuration of hyperparameters.
    Parameters:
        config : A dictionary containing the hyperparameters for the model.
        x_train : The training dataset as a PyTorch tensor.
        x_test : The testing dataset as a PyTorch tensor.
        raw_data : The original coordinates of the dataset after removing rigid-body motion.
        sequence_names : The sequence names corresponding to each frame in the dataset.
        scaler : An instance of a scaler (Standardizer or MinMaxScaler) used for data normalization, or None if no scaling is applied.
        run_id : A unique identifier for the current run, used for saving results.
        output_dir : The directory where the results will be saved.
        lr_step_size : The step size for the learning rate scheduler (default: 100).
        lr_gamma : The gamma value for the learning rate scheduler (default: 0.5).
    Returns:
        metrics : A dictionary containing the evaluation metrics for the current configuration, including final training loss, final testing loss, RMSD, mean RMSD, standard deviation of RMSD, and the hyperparameters used.
    """
    original_dim = x_train.shape[1]
    n_frames = x_train.shape[0] + x_test.shape[0]

    if config["HIDDEN_DIMS"] is None:
        activation = activations[config["activation"]]
        print(activation)
        model, optimizer, scheduler = build_autoencoder(original_dim=original_dim, latent_dim=config["LATENT_DIM"], learning_rate=config["LEARNING_RATE"], 
                                                        device=DEVICE, lr_step_size=lr_step_size, lr_gamma=lr_gamma, activation=activation)
    else:
        activation = activations[config["activation"]]
        model, optimizer, scheduler = build_autoencoder(original_dim=original_dim, latent_dim=config["LATENT_DIM"], hidden_dims=config["HIDDEN_DIMS"],
                                             learning_rate=config["LEARNING_RATE"], device=DEVICE, lr_step_size=lr_step_size, lr_gamma=lr_gamma, activation=activation)

    history, history_test = train_autoencoder(model, optimizer, x_train, x_test, epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"],
                                              device=DEVICE, scheduler=scheduler)

    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_samples = raw_data.shape[0]
    n_atoms = raw_data.shape[1]
    batch_size = config["BATCH_SIZE"]
    z_chunks = []
    rmsd_chunks = []
    rmsd_chunks_unscaled = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_raw = raw_data[start:end]
            batch_flat = torch.tensor(flatten_ensemble(batch_raw), dtype=torch.float32).to(DEVICE)
            z, reconstructed = model(batch_flat)
            z_chunks.append(z.cpu().numpy())
            ## unflatten the reconstructed data to shape (batch_size, n_atoms, n_features) and remove the padding indicator before inverse transforming
            unflattened_reconstructed = unflatten_ensemble(reconstructed.cpu().numpy(), n_atoms)

            batch_rmsd = RMSD(batch_raw[:,:,:3], unflattened_reconstructed[:,:,:3])
            rmsd_chunks.append(batch_rmsd)

            if scaler is not None:
                unflattened_reconstructed_unscaled = scaler.inverse_transform(torch.as_tensor(unflattened_reconstructed[:,:,:3], dtype=torch.float32))
                unflattened_raw_unscaled = scaler.inverse_transform(torch.as_tensor(batch_raw[:,:,:3], dtype=torch.float32))
                batch_rmsd_unscaled = RMSD(unflattened_raw_unscaled, unflattened_reconstructed_unscaled)
                rmsd_chunks_unscaled.append(batch_rmsd_unscaled)
            

            del batch_flat, z, reconstructed
            if hasattr(DEVICE, "type") and DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    z_latent = np.concatenate(z_chunks, axis=0)
    # save z_latent to a file with sequence names as csv based on latent dimension
    z_latent_df = pd.DataFrame(z_latent, columns=[f"z_{i}" for i in range(config["LATENT_DIM"])])
    z_latent_df["sequence_names"] = sequence_names

    z_latent_df.to_csv(f"{run_dir}/z_latent.csv", index=False)

    rmsd = np.concatenate(rmsd_chunks, axis=0)
    if scaler:
        rmsd_unscaled = np.concatenate(rmsd_chunks_unscaled, axis=0)
    else:
        rmsd_unscaled = rmsd

    # get z_latent for all frames and plot it with color-coded sequence names

    plot_latent_space(z_latent, color=sequence_names, color_label="sequence_names", save_path=f"{run_dir}/latent_space.png")
    print("Saved latent_space.png")
    plot_losses(history, history_test, save_path=f"{run_dir}/loss_curves.png")
    print("Saved loss_curves.png")

    metrics = {"run_id": run_id, "final_train_loss": history[-1], "final_test_loss": history_test[-1], "RMSD": rmsd, "rmsd_mean": np.mean(rmsd), "rmsd_std": np.std(rmsd),
               "RMSD_unscaled": rmsd_unscaled, "rmsd_mean_unscaled": np.mean(rmsd_unscaled), "rmsd_std_unscaled": np.std(rmsd_unscaled), **config}

    with open(f"{run_dir}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # save the model
    torch.save(model.state_dict(), f"{run_dir}/model.pth")
    return metrics


def grid_search(x_train, x_test, raw_data, sequence_names, scaler, hyperparams_dict, output_dir, add_padding_indicator, lr_step_size=100, lr_gamma=0.5):
    """
    Perform a grid search over hyperparameter combinations for training and evaluating an autoencoder model.
    Parameters:
        x_train : The training dataset as a PyTorch tensor.
        x_test : The testing dataset as a PyTorch tensor.
        raw_data : The original coordinates of the dataset after removing rigid-body motion.
        sequence_names : The sequence names corresponding to each frame in the dataset.
        scaler : An instance of a scaler (Standardizer or MinMaxScaler) used for data normalization, or None if no scaling is applied.
        hyperparams_dict : A dictionary containing lists of hyperparameter values to search over.
        output_dir : The directory where the results will be saved.
        lr_step_size : The step size for the learning rate scheduler (default: 100).
        lr_gamma : The gamma value for the learning rate scheduler (default: 0.5).
    Returns:
        results : A list of dictionaries containing the evaluation metrics for each hyperparameter configuration, sorted by mean RMSD in ascending order.
    """
    hyperparams_combinations = gen_parms_combinations(**hyperparams_dict)
    print(f"Running grid search over {len(hyperparams_combinations)} configurations...")

    results = []
    for i, config in enumerate(hyperparams_combinations):
        run_id = f"run_{i + 1:03d}_ld{config['LATENT_DIM']}_layer{config['NUM_HIDDEN_LAYER']}_B{config['BATCH_SIZE']}_E{config['EPOCHS']}_lr{config['LEARNING_RATE']}_act{config['activation']}"
        print(f"\n[{i + 1}/{len(config)}], config={config}")
 
        metrics = run_single_config(config, x_train, x_test, raw_data, sequence_names, scaler, run_id, output_dir, add_padding_indicator,
                                    lr_step_size=lr_step_size, lr_gamma=lr_gamma)
        results.append(metrics)
        print(f"  -> train_loss={metrics['final_train_loss']:.4f}  rmsd_mean={metrics['rmsd_mean']:.4f}, rmsd_std={metrics['rmsd_std']:.4f}, rmsd_mean_unscaled={metrics['rmsd_mean_unscaled']:.4f}, rmsd_std_unscaled={metrics['rmsd_std_unscaled']:.4f}")

    results.sort(key=lambda m: m["rmsd_mean"])
    return results

hyperparams_dict = {
    "BATCH_SIZE": args.BATCH_SIZE,
    "LATENT_DIM": args.LATENT_DIM,
    "NUM_HIDDEN_LAYER": args.NUM_HIDDEN_LAYER,
    "EPOCHS": args.EPOCHS,
    "LEARNING_RATE": args.LEARNING_RATE,
    "HIDDEN_DIMS": args.HIDDEN_DIMS,
    "activation": args.activation
}

x_train, x_test, raw_data, sequence_names, scaler = load_dataset(args.output_dir, args.save_file_name, float(args.train_test_split), args.add_padding_indicator, args.scalar_type)

results = grid_search(x_train, x_test, raw_data, sequence_names, scaler, hyperparams_dict, args.output_dir, add_padding_indicator=args.add_padding_indicator,
                      lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma)

summary_path = f"{args.output_dir}/grid_search_summary.pkl"

with open(summary_path, "wb") as f:
    pickle.dump(results, f)

best = results[0]
print("\nBest configuration (lowest RMSD):")
print(f"  -> run_id={best['run_id']}, train_loss={best['final_train_loss']:.4f}, rmsd_mean={best['rmsd_mean']:.4f}, rmsd_std={best['rmsd_std']:.4f}, rmsd_mean_unscaled={best['rmsd_mean_unscaled']:.4f}, rmsd_std_unscaled={best['rmsd_std_unscaled']:.4f}")   
