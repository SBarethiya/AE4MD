import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT
from MDAnalysis.analysis import align
import numpy as np
import argparse, os, pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional, Tuple

parser = argparse.ArgumentParser(description="Input preparation of the model for protein dynamics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 fromfile_prefix_chars="@")

parser.add_argument("--input_list", help="input list (format: txt)", required=True)
parser.add_argument("--input_dir", help="input directory (format: dir)", required=True)
parser.add_argument("--output_dir", help="output directory (format: dir)", required=True)
parser.add_argument("--save_file_name", help="file name to save the output (format: str)", required=True)

parser.add_argument("--padding", help="padding method (format: str)", required=False, default="False")
parser.add_argument("--align_trajectory", help="input list (format: txt)", required=False, default="False")
parser.add_argument("--align_selection", help="selection for alignment (format: str)", required=False, default="None")
parser.add_argument("--input_selection", help="selection for input mainly the part of protein otherwise type is only required (format: str)", required=False, default="None")
parser.add_argument("--type", help="type of atoms to select (CA, backbone, heavy, dihedral, internal_coordinates)", required=False, default="CA")
parser.add_argument("--n_frame", help="number of frames to select (format: int)", required=False, default="None")
parser.add_argument("--start", help="starting frame index (format: int)", required=False, default=0)
parser.add_argument("--n_workers", help="number of worker processes (format: int, default: all CPUs)", required=False, type=int, default=os.cpu_count())
args = parser.parse_args()

def align_trajectoryies(psf, traj, selection="protein"):
    """
    -- Align trajectory to reference structure
    Input: 
        psf: topology file
        pdb: reference structure file
        traj: trajectory file
        selection: selection string for MDAnalysis (default is protein)
    Output:
        u: MDAnalysis Universe object with aligned trajectory
    """

    u = mda.Universe(psf, traj)
    ref = mda.Universe(psf, traj)
    protein = u.select_atoms(selection)
    ref_protein = ref.select_atoms(selection)
    aligner = align.AlignTraj(u, ref, select=selection, in_memory=True)
    aligner.run()
    return u


def get_xyz(u, type, input_selection, start, n_frame):
    """
    -- Get the xyz coordinates of the selected atoms from the trajectory
    Input: 
        psf: topology file
        xtc: trajectory file
        type: type of atoms to select (CA, backbone, heavy)
        selection: selection string for MDAnalysis (default is None)
    Output:
        xyz_array: numpy array of shape (n_frames, n_atoms, 3) containing the xyz coordinates of the selected atoms
    """
    if type == "CA":
        selection = "name CA"
    elif type == "backbone":
        selection = "backbone and not name H*"
    elif type == "heavy":
        selection = "not name H*"
    else:
        selection = "protein"

    if input_selection != "None":
        selection = selection + " and " + input_selection

    select_atoms = u.select_atoms(selection)
    selected_indices = np.linspace(start, len(u.trajectory)-1, n_frame, dtype=int)

    if type == "dihedral" or type=="internal_coordinates":
        if selection == "None":
            selection = "backbone and not name H*"
            dihedral_atoms = select_atoms.select_atoms(f"protein and {selection}")
        dihedral_atoms = select_atoms.select_atoms(f"protein and {selection}")
        select_atoms = dihedral_atoms
        R = BAT(selection)
        R.run()

        # split Ic to get bonds, angles and dihedrals
        # Number of frames form selected_indices
        bonds = R.results.bat[:, 9:len(u.atoms)-3+9]
        angles = R.results.bat[:, len(u.atoms)-3+9:(len(u.atoms)-3)*2+9]
        dihedrals = R.results.bat[:, (len(u.atoms)-3)*2+9:]
        if type == "dihedral":
            xyz_array = dihedrals
        elif type == "internal_coordinates":
            xyz_array = np.concatenate((bonds, angles, dihedrals), axis=1)
    else:
        xyz = []
        print(u.trajectory[selected_indices])   
        for ts in u.trajectory[selected_indices]:
            xyz.append(select_atoms.positions)
        xyz_array = np.array(xyz)

    return xyz_array

def resolve_topology(input_dir: str, name: str) -> Optional[str]:
    """Find the .psf topology file for ``name``, or None if not found."""
    if os.path.exists(f"{input_dir}/{name}.psf"):
        return f"{input_dir}/{name}.psf"
    if os.path.isdir(f"{input_dir}/{name}"):
        psf_files = [f for f in os.listdir(f"{input_dir}/{name}") if f.endswith(".psf")]
        print(f"Found {psf_files} psf files in {input_dir}/{name}")
        if len(psf_files) > 0:
            return f"{input_dir}/{name}/{psf_files[0]}"
    return None
 
 
def resolve_trajectory(input_dir: str, name: str) -> Optional[str]:
    """Find the .xtc/.dcd trajectory file for ``name``, or None if not found."""
    if os.path.exists(f"{input_dir}/{name}.xtc"):
        return f"{input_dir}/{name}.xtc"
    if os.path.exists(f"{input_dir}/{name}.dcd"):
        return f"{input_dir}/{name}.dcd"
    if os.path.isdir(f"{input_dir}/{name}"):
        xtc_files = [f for f in os.listdir(f"{input_dir}/{name}") if f.endswith(".xtc")]
        dcd_files = [f for f in os.listdir(f"{input_dir}/{name}") if f.endswith(".dcd")]
        if len(xtc_files) > 0:
            return f"{input_dir}/{name}/{xtc_files[0]}"
        if len(dcd_files) > 0:
            return f"{input_dir}/{name}/{dcd_files[0]}"
    return None

def process_one_input(
    idx_name: Tuple[int, str],
    input_dir: str,
    align_selection: str,
    input_selection: str,
    atom_type: str,
    start: int,
    n_frame_arg: str,
) -> Tuple[int, Optional[np.ndarray]]:
    """Load one system, extract its xyz (or internal-coordinate) features, and
    tag every atom/frame entry with ``idx`` in a new trailing channel.
 
    Runs in a worker process — everything it needs (paths, selection
    strings) is passed in as plain picklable arguments; the MDAnalysis
    Universe itself is created fresh inside the worker.
 
    Returns
    -------
    (idx, xyz_array) or (idx, None) if the system had to be skipped.
    """
    idx, name = idx_name
    print(f"Processing {name}...")
 
    psf = resolve_topology(input_dir, name)
    if psf is None:
        print(f"WARNING: No topology file found for {name}. Skipping.")
        return idx, None
 
    xtc = resolve_trajectory(input_dir, name)
    if xtc is None:
        print(f"WARNING: No trajectory file found for {name}. Skipping.")
        return idx, None
 
    if align_selection != "None":
        u = align_trajectoryies(psf, xtc, align_selection)
    else:
        u = mda.Universe(psf, xtc)
 
    n_frame = len(u.trajectory) if n_frame_arg == "None" else int(n_frame_arg)
 
    xyz_array = get_xyz(u, atom_type, input_selection, start, n_frame)
    # tag every entry with the system index in a new trailing channel
    xyz_array = np.concatenate(
        (xyz_array, np.full((xyz_array.shape[0], xyz_array.shape[1], 1), idx)),
        axis=2,
    )
    print(f"xyz_array shape: {xyz_array.shape}")
    return idx, xyz_array

input_list = np.loadtxt(args.input_list, dtype=str)
input_dir = args.input_dir
output_dir = args.output_dir
file_name = args.save_file_name

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    output_file = f"{output_dir}/{file_name}.npy"
    if os.path.exists(output_file):
        while True:
            response = input(
                f"WARNING: {output_file} already exists.\n"
                "Overwrite it? [y/n]: "
            ).strip().lower()

            if response in ("y", "yes"):
                break  # continue using output_file

            elif response in ("n", "no"):
                new_name = input(
                    "Enter a new output filename (or press Enter to cancel): "
                ).strip()

                if not new_name:
                    raise SystemExit("Operation cancelled.")

                output_file = os.path.join(output_dir, new_name)
                break

            else:
                print("Please enter 'y' or 'n'.")

worker = partial(
    process_one_input,
    input_dir=input_dir,
    align_selection=args.align_selection,
    input_selection=args.input_selection,
    atom_type=args.type,
    start=int(args.start),
    n_frame_arg=args.n_frame,
)

with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
    results = list(executor.map(worker, enumerate(input_list)))
 
final_array = []
sequence_lengths = []
for idx, xyz_array in results:
    if xyz_array is None:
        continue
    final_array.extend(xyz_array)
    sequence_lengths.append(xyz_array.shape[1])

if len(set(sequence_lengths)) == 1:
    print("\nAll sequences have the same length. \n final_array shape: ", np.array(final_array).shape)
else:
    if args.padding == "False":
        print("\nWARNING: All sequences have different lengths.")
        print("Padding is False, please make sure batches are of the same length...\n")
    else:
        print("\nWARNING: All sequences have different lengths.")
        print("Sequences have different lengths. Padding with zeros to the maximum length.\n")
        max_length = max(sequence_lengths)
        padded_array = []
        for seq in final_array:
            if seq.shape[0] != max_length:
                padding = np.zeros((max_length - seq.shape[0], seq.shape[1]))
                padded_seq = np.concatenate((seq, padding), axis=0)
                padded_array.append(padded_seq)
            else:
                padded_array.append(seq)
        final_array = padded_array

print(f"\nSaving final_array to {output_dir}/{file_name}.pkl with shape {np.array(final_array).shape}...")
with open(f"{output_dir}/{file_name}.pkl", "wb") as f:
    pickle.dump(final_array, f)
