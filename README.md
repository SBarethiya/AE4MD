# MD Conformational Ensemble Autoencoder

A PyTorch autoencoder pipeline for visualizing a molecular dynamics (MD) conformational ensemble in 2D. Frames are superposed to remove rigid-body motion, flattened and scaled, then compressed to a 2-D latent space by an autoencoder whose depth/width is fully hyperparameter-driven. A grid-search is utlized for training and ranking multiple configurations by reconstruction RMSD.

## Project structure

```
.
├── main.py                      # CLI entry point: load data, grid-search, rank configs
├── scripts/
│   └── Input_generation.py          # Extract xyz/features from raw MD trajectories (parallel, per-system)
|   └── model.py         # Autoencoder architecture + MD preprocessing + plotting
├── requirements.txt
└── README.md
```

There are two separate stages: `Input_generation.py` turns raw trajectory files (topology + trajectory, one pair per system) into a single coordinate array, and `main.py` consumes that array to train and evaluate autoencoders. See *Known limitations* below for a filename/shape mismatch between the two that you'll need to bridge by hand for now.

## Environment setup
### conda

```bash
conda create -n md-ae python=3.10
conda activate md-ae
pip install -r requirements.txt
```

GPU is optional — `main.py` auto-selects `cuda` if `torch.cuda.is_available()`, otherwise falls back to `cpu`. `Input_generation.py` parallelizes across CPU processes (`--n_workers`, default: all cores) and doesn't use the GPU. 

## Stage 1: extract coordinates with `Input_generation.py`

Turns one or more raw trajectories into a single coordinate array, run in parallel across systems (one process per system, via `--n_workers`).

### Expected input layout

A text file (`--input_list`) with one system name per line:

```
sysA
sysB
```

For each name, `--input_dir` must contain either `{name}.psf` + (`{name}.xtc` or `{name}.dcd`), or a subdirectory `{name}/` containing one `.psf` and one
`.xtc`/`.dcd` file.

### Example

```bash
python Input_generation.py \
  --input_list systems.txt \
  --input_dir /path/to/trajectories \
  --output_dir prepared \
  --save_file_name my_traj \
  --type CA \
  --start 500
  --n_frame 1000 \
  --align_selection "protein" \
  --n_workers 4
  --add_padding_indicator True
```

### Key arguments

| Argument             | Format | Description                                                                |
|----------------------|--------|----------------------------------------------------------------------------|
| `--input_list`       | path   | Text file listing one system name per line                                 |
| `--input_dir`        | path   | Directory containing each system's topology/trajectory files               |
| `--output_dir`       | path   | Where the output `.pkl` is written                                         |
| `--save_file_name`   | str    | Output is written to `{output_dir}/{save_file_name}.pkl`                   |
| `--type`             | str    | Atom selection: `CA`, `backbone`, `heavy`, `dihedral`, `internal_coordinates` (else full `protein`) |
| `--align_selection`  | str    | MDAnalysis selection string to align each trajectory on before extraction; `"None"` skips alignment |
| `--input_selection`  | str    | Extra MDAnalysis selection ANDed onto the atom-type selection                |
| `--n_frame`          | int    | Number of frames to sample per system (evenly spaced); `"None"` = all frames |
| `--start`            | int    | Starting frame index for sampling                                            |
| `--padding`          | str    | If systems differ in atom count, pad with zeros (`"True"`) or leave as-is (`"False"`, default) |
| `--n_workers`        | int    | Number of parallel worker processes (default: all CPU cores)                 |
| `--add_padding_indicator` | bool | dd a padding indicator column (1 for padded, 0 for not padded)            |

Each frame is tagged with an extra trailing channel holding its system index (0 for the first line in `--input_list`, 1 for the second, etc.), so the
output array has shape `(n_frames_total, n_atoms, 4)` — not `(n_frames, n_atoms, 3)`. See *Known limitations* for how this needs to be reconciled with what `main.py` expects.

## Stage 2: train/evaluate autoencoders with `main.py`

`main.py` expects a pickled NumPy array at `{output_dir}/{save_file_name}_xyz.pkl` containing the coordinate ensemble:

- Shape: `(n_frames, n_atoms, 3)`
- All frames must currently have the **same** `n_atoms`, with atom `i` in every frame referring to the same physical atom (same order/identity), since alignment and flattening assume a fixed, consistent atom mapping.

### Usage

```bash
python main.py \
  --output_dir runs \
  --save_file_name my_traj \
  --BATCH_SIZE "[32, 64]" \
  --LATENT_DIM [2] \
  --NUM_HIDDEN_LAYER "[2, 4]" \
  --HIDDEN_DIMS "[[64, 128], [32, 64, 128]]" \
  --LEARNING_RATE "[0.001, 0.0001]" \
  --EPOCHS "[100, 200]" \
  --scalar_type standard \
  --seed 42 \
  --lr_step_size 100 \
  --lr_gamma 0.5 \
  --activation relu
```

### Key arguments

| Argument          | Format         | Description                                                             |
|-------------------|----------------|-----------------------------------------------------------------------  |
| `--output_dir`    | str            | Directory containing the input `.pkl` and where run outputs are written |
| `--save_file_name`| str            | Base filename (expects `{save_file_name}_xyz.pkl`)                      |
| `--BATCH_SIZE`    | list[int]      | Batch sizes to sweep                                                    |
| `--LATENT_DIM`    | list[int]      | Latent dimensions to sweep (use `2` for direct visualization)           |
| `--NUM_HIDDEN_LAYER` | list[int]   | Number of hidden layers in both encoder and decoder                     |
| `--HIDDEN_DIMS`   | list[list[int]]| Architectures to sweep — each inner list is one architecture's hidden-layer widths, e.g. `[[64,128]]` = one hidden-layer-64-then-128 network |
| `--LEARNING_RATE` | list[float]    | Learning rates to sweep                                                 |
| `--EPOCHS`        | list[int]      | Epoch counts to sweep                                                   |
| `--scalar_type`   | `standard` \| `minmax` \| `none` | Feature scaling applied before training               |
| `--seed`          | int            | Random seed for the train/test split and model init                     |
| `--lr_step_size`  | int            | Number of epochs between learning-rate scheduler steps                  |
| `--lr_gamma`      | float          | Multiplicative factor applied by the learning-rate scheduler            |
| `--activation`    | str            | Activation function used by the autoencoder; default relu               |

List-valued arguments must be valid Python list syntax and quoted so your
shell passes them through intact:

```bash
--HIDDEN_DIMS "[[100, 100]]"          # one architecture
--HIDDEN_DIMS "[[100, 100], [50, 100, 150]]"   # sweep two architectures
```

## Pipeline

1. **`kabsch_align`** — superpose every frame onto a reference structure, removing translation/rotation so the AE learns conformational change, not rigid-body drift.
2. **`flatten_ensemble`** — reshape `(n_frames, n_atoms, 3)` → `(n_frames, n_atoms*3)`.
3. **Scaling** — `Standardizer` (zero-mean/unit-variance) or `MinMaxScaler` (scales to `[0, 1]`), fit on the training split only.
4. **`build_autoencoder` / `train_autoencoder`** — MSE-reconstruction training with an Adam optimizer.
5. **Grid search** — every combination of `BATCH_SIZE`, `LATENT_DIM`, `HIDDEN_DIMS`, `LEARNING_RATE`, `EPOCHS` is trained and evaluated.
6. **Ranking** — configurations are sorted by mean reconstruction RMSD (original aligned frames vs. decoded reconstructions, unscaled back to real coordinate units).

## Outputs

For each grid-search configuration, `main.py` writes to `{output_dir}/{run_id}/`:

- `latent_space.png` — 2D latent embedding, colored by frame index
- `loss_curves.png` — train/validation MSE loss per epoch
- `metrics.pkl` — final losses, mean reconstruction RMSD, and the config used

A `grid_search_summary.pkl` in `{output_dir}` collects every run's metrics, sorted by RMSD (best first), and the best configuration is printed at the end.

## Known limitations

- **All frames must share the same atom count and atom ordering** for `main.py`'s alignment/flattening step. Variable atom counts across systems (e.g. unresolved residues, or ensembles spanning different sequences) are not yet supported end-to-end — see the padding note below.
- **`Input_generation.py` output doesn't directly match what `main.py` expects**: 
  - Filename: `Input_generation.py` writes `{save_file_name}.pkl`, but `main.py` reads `{save_file_name}_xyz.pkl`. You'll need to rename/copy the file (or adjust one of the two scripts) between stages for now.
  - Shape: `Input_generation.py`'s output has a trailing system-index channel, shape `(n_frames, n_atoms, 4)`, while `main.py` expects raw `(n_frames,  n_atoms, 3)` coordinates. Drop the last channel (and decide how to handle multiple systems, e.g. train per-system or concatenate) before handing the array to `main.py`.
- **Padding-axis mismatch in `Input_generation.py`**: when systems have different atom counts, `sequence_lengths` records `xyz_array.shape[1]` (atom count),but the padding step pads/compares along `seq.shape[0]` (frame count) against that same `max_length` value — those are different axes, so `--padding True` likely doesn't pad what it's intended to when atom counts (not frame counts) differ across systems. Flagging this rather than guessing at the  intended fix — let us know which axis should actually be padded and we can correct it.
- `--type` (atom selection) is applied in `Input_generation.py`, but `main.py`'s own `--type` argument is still accepted and unused — it's a no-op there  since atom selection now happens upstream, in `Input_generation.py`.