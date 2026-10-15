import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.memory import MemoryReader

import numpy as np
import pandas as pd
import argparse, os, pickle, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, Tuple

parser = argparse.ArgumentParser(description="Input preparation of the model for protein dynamics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 fromfile_prefix_chars="@")

parser.add_argument("--input_list", help="input list (format: txt)", required=True)
parser.add_argument("--input_dir", nargs="+", help="input directory (format: dir)", required=True)
parser.add_argument("--output_dir", help="output directory (format: dir)", required=True)
parser.add_argument("--save_file_name", help="file name to save the output (format: str)", required=True)
parser.add_argument("--padding", help="padding method (format: str)", required=False, default="False")
parser.add_argument("--align_selection", help="selection for alignment (format: str)", required=False, default="None")
parser.add_argument("--input_selection", help="selection for input mainly the part of protein otherwise type is only required (format: str)", required=False, default="None")
parser.add_argument("--type", help="type of atoms to select (CA, backbone, heavy, dihedral, internal_coordinates)", required=False, default="CA")
parser.add_argument("--n_frame", help="number of frames to select (format: int)", required=False, default="None")
parser.add_argument("--start", help="starting frame index (format: int)", required=False, default=0)
parser.add_argument("--start_residue", help="starting residue index (format: int)", required=False)
parser.add_argument("--end_residue", help="ending residue index (format: int)", required=False)
parser.add_argument("--n_workers", help="number of worker processes (format: int, default: all CPUs)", required=False, type=int, default=os.cpu_count())
parser.add_argument("--add_padding_indicator", help="Add a padding indicator column (1 for padded, 0 for not padded)", action="store_true", default=False)
args = parser.parse_args()


def subsample_to_memory(u, selected_indices):
    """
    Subsample the trajectory to only include the specified frames and load it into memory.
    """
    frames = np.asarray(selected_indices)
    n_atoms = len(u.atoms)
    coords = np.empty((len(frames), n_atoms, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        u.trajectory[int(f)]
        coords[i] = u.atoms.positions
    u.load_new(coords, format=MemoryReader)
    return u


def match_reference_chains(mobile_atoms, ref_positions_all, ref_segids, unique_ref_segids):
    """
    Match the chains in the mobile system to a subset of chains in the reference system.
    """
    mobile_segids = sorted(set(mobile_atoms.segids))
    n_chains_mobile = len(mobile_segids)
 
    if n_chains_mobile > len(unique_ref_segids):
        raise ValueError(f"System has {n_chains_mobile} chains, more than the reference's", f"{len(unique_ref_segids)} chains -- can't pick a matching subset.")
 
    chosen_ref_segids = set(unique_ref_segids[n_chains_mobile:])
    mask = np.isin(ref_segids, list(chosen_ref_segids))
    ref_positions = ref_positions_all[mask]
    return ref_positions


# aligning each trajectory input to its frame 0 so they will all be in the same orientation
def align_trajectories(u, ref_positions, idx, selection):
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
    # Both universes start on frame 0 by default, so ref is already at the
    # frame we want to align onto. Align mobile (u) onto ref directly.
    if ref_positions is None:
        return u
 
    mobile_atoms = u.select_atoms(selection)
    ref_frame_universe = mda.Merge(mobile_atoms)
    ref_frame_universe.load_new(ref_positions[np.newaxis, :, :], order="fac")
 
    align.AlignTraj(u, ref_frame_universe, select=selection, in_memory=True,).run()
    # Save the aligned trajectory to a new file only 10 frames as PDB with selection
    # with mda.Writer(f"{args.output_dir}/{idx}_aligned.pdb", n_atoms=len(mobile_atoms)) as W:
    #     for ts in u.trajectory[:10]:
    #         W.write(mobile_atoms)
    return u


def get_xyz(u, type, input_selection):
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
    # selecting CA or backbone atom beside H
    if type == "CA":
        selection = "name CA"
    elif type == "backbone":
        selection = "backbone and not name H*"
    elif type == "heavy":
        selection = "not name H*"
    else:
        selection = "protein"

    # you can specify input selection like just core or something
    if input_selection != "None":
        selection = selection + " and " + input_selection

    # frame number was specified in input so grabs the atoms from those
    select_atoms = u.select_atoms(selection)

    # grabs the coordinates for those frame/atom
    if type == "dihedral" or type=="internal_coordinates":
        if selection == "None":
            selection = "backbone and not name H*"

        select_atoms = select_atoms.select_atoms(f"protein and {selection}")

        R = BAT(select_atoms)
        R.run()

        # split Ic to get bonds, angles and dihedrals
        # Number of frames form selected_indices
        n_sel = len(select_atoms)
        bonds = R.results.bat[:, 9:n_sel-3+9]
        angles = R.results.bat[:, n_sel-3+9:(n_sel-3)*2+9]
        dihedrals = R.results.bat[:, (n_sel-3)*2+9:]
        if type == "dihedral":
            xyz_array = dihedrals
        elif type == "internal_coordinates":
            xyz_array = np.concatenate((bonds, angles, dihedrals), axis=1)
    else:
        xyz = []
        for ts in u.trajectory:
            xyz.append(select_atoms.positions)
        xyz_array = np.array(xyz)
    return xyz_array

# finding psf file in directory
def resolve_topology(input_dirs: list, name: str) -> Optional[str]:
    """Find the .psf topology file for ``name``, or None if not found."""
    for input_dir in input_dirs:
        if os.path.exists(f"{input_dir}/{name}.psf"):
            return f"{input_dir}/{name}.psf"
        if os.path.isdir(f"{input_dir}/{name}"):
            psf_files = [f for f in os.listdir(f"{input_dir}/{name}") if f.endswith(".psf")]
            if len(psf_files) > 0:
                return f"{input_dir}/{name}/{psf_files[0]}"
    return None
 
# finding dcd file in directory 
def resolve_trajectory(input_dirs: list, name: str) -> Optional[str]:
    """Find the .xtc/.dcd trajectory file for ``name``, or None if not found."""
    for input_dir in input_dirs:
        if os.path.exists(f"{input_dir}/{name}.xtc"):
            return f"{input_dir}/{name}.xtc"
        if os.path.exists(f"{input_dir}/{name}/{name}_hyres_combined.dcd"):
            return f"{input_dir}/{name}/{name}_hyres_combined.dcd"
        if os.path.isdir(f"{input_dir}/{name}"):
            xtc_files = [f for f in os.listdir(f"{input_dir}/{name}") if f.endswith(".xtc")]
            dcd_files = [f for f in os.listdir(f"{input_dir}/{name}") if f.endswith(".dcd")]
            if len(xtc_files) > 0:
                return f"{input_dir}/{name}/{xtc_files[0]}"
            if len(dcd_files) > 0:
                return f"{input_dir}/{name}/{dcd_files[0]}"
    print(f"WARNING: No trajectory file found for {name}. Skipping.")
    return None

def process_one_input(
    idx_name: pd.DataFrame,
    idx_start:  pd.DataFrame,
    idx_end: pd.DataFrame,
    segid: int,
    ref_positions_all: Optional[np.ndarray],
    ref_segids: Optional[np.ndarray],
    unique_ref_segids: Optional[list],
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
    _, resid_start = idx_start
    _, resid_end = idx_end
    _, segid = segid

    print(f"Processing {name}...")
    #find psf
    psf = resolve_topology(input_dir, name)
    if psf is None:
        print(f"WARNING: No topology file found for {name}. Skipping.")
        return idx, None
    # find dcd
    xtc = resolve_trajectory(input_dir, name)
    if xtc is None:
        print(f"WARNING: No trajectory file found for {name}. Skipping.")
        return idx, None
    
    # call alignment function (align translation/rotation)
    try:
        u = mda.Universe(psf, xtc)
    except Exception as e:
        print(f"ERROR: Failed to load Universe for {name}. Skipping. Error: {e}")
        return idx, None
    
    n_frame = len(u.trajectory) if n_frame_arg == "None" else int(n_frame_arg)
    if len(u.trajectory) < start or len(u.trajectory) < int(start+n_frame):
        print(f"WARNING: n_frame ({len(u.trajectory)}) is less than start ({start}) for {name}. Skipping.")
        return idx, None
    
    selected_indices = np.linspace(start, len(u.trajectory) - 1, n_frame, dtype=int)
    u = subsample_to_memory(u, selected_indices)
    if align_selection:
        align_selection_final = align_selection + " and segid " + str(segid)
        mobile_atoms_for_matching = u.select_atoms(align_selection_final)
        if mobile_atoms_for_matching.n_atoms != len(ref_positions_all) or ref_segids is not None:
            this_ref_positions = match_reference_chains(mobile_atoms_for_matching, ref_positions_all, ref_segids, unique_ref_segids)
        else: 
            this_ref_positions = ref_positions_all
        u = align_trajectories(u, this_ref_positions, idx, align_selection_final)
        # pull the xyz coordinates for selected atom/frame
        if resid_start:
            resid_selection = f"resid {resid_start} to {resid_end}"
            if input_selection != "None":
                resid_selection = resid_selection + " and " + input_selection + " and segid " + str(segid) 
        else:
            if input_selection != "None":
                resid_selection = input_selection + " and segid " + str(segid) 
            else:
                resid_selection = "protein"
        xyz_array = get_xyz(u, atom_type, resid_selection)
        # tag every entry with the system index in a new trailing channel
        xyz_array = np.concatenate((xyz_array, np.full((xyz_array.shape[0], xyz_array.shape[1], 1), idx)), axis=2,)

    return idx, xyz_array

# input_list = np.loadtxt(args.input_list, dtype=str)
input_list = pd.read_csv(args.input_list, names=["Name", "Start", "End", "Missing", "Segid"], header=None)
print(input_list)
print(f"Total number of sequences in the file: {len(input_list)}")

input_dir = args.input_dir
output_dir = args.output_dir
file_name = args.save_file_name

# check if reference files exist
psf_ref = resolve_topology(input_dir, "6osj") 
traj_ref = resolve_trajectory(input_dir, "6osj")
if args.align_selection != "None":
    u_ref = mda.Universe(psf_ref, traj_ref)
    selected_indices = np.linspace(len(args.start), len(u_ref.trajectory) - 1, len(args.n_frame), dtype=int)
    u_ref = subsample_to_memory(u_ref, selected_indices)
    u_ref.trajectory[0]
    final_align_selection = args.align_selection + " and segid " + str(input_list["Segid"][0])
    print(f"Aligning to reference structure using selection: {final_align_selection}")
    ref_atoms_all = u_ref.select_atoms(final_align_selection)
    print(f"Reference atoms selected for alignment: {ref_atoms_all.n_atoms} atoms")
    ref_positions_all = u_ref.select_atoms(final_align_selection).positions.copy()

    # check if there are multiple segids in the selection
    if len(ref_atoms_all.segments) > 1:        
        ref_segids = np.array(u_ref.segments)
        unique_ref_segids = sorted(set(ref_segids))
        for segid in unique_ref_segids:
            seg_atoms = ref_atoms_all.select_atoms(f"segid {segid}")
            seg_positions = seg_atoms.positions.copy()
            seg_positions -= seg_positions.mean(axis=0)
            ref_positions_all[seg_atoms.indices - ref_atoms_all.indices.min()] = seg_positions
    else:
        ref_segids = None
        unique_ref_segids = None

    del u_ref
else:
    ref_positions = None

worker = partial(
    process_one_input,
    ref_positions_all=ref_positions_all,
    ref_segids=ref_segids,
    unique_ref_segids=unique_ref_segids,
    input_dir=input_dir,
    align_selection=args.align_selection,
    input_selection=args.input_selection,
    atom_type=args.type,
    start=int(args.start),
    n_frame_arg=args.n_frame,
)

if args.start_residue is not None and args.end_residue is not None:
    input_list["Start"] = args.start_residue
    input_list["End"] = args.end_residue

# Do multiprocessing to process each input in parallel and if any of the input fails, it will be skipped and the rest will be processed
with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
    results = list(executor.map(worker, enumerate(input_list["Name"]), enumerate(input_list["Start"]), enumerate(input_list["End"]), enumerate(input_list["Segid"])))

final_array = []
sequence_lengths = []
for idx, xyz_array in results:
    print(xyz_array.shape)
    if xyz_array is None:
        continue
    final_array.extend(xyz_array)
    sequence_lengths.append(xyz_array.shape[1])

print(f"\nSaving raw array to {output_dir}/{file_name}_raw.pkl...")
with open(f"{output_dir}/{file_name}_raw.pkl", "wb") as f:
    pickle.dump(final_array, f)


if len(set(sequence_lengths)) == 1:
    print("\nAll sequences have the same length. \n final_array shape: ", np.array(final_array).shape)
    padded_array = None

else:
    if args.padding == "False":
        print("\nWARNING: All sequences have different lengths.")
        print("Padding is False, please make sure batches are of the same length...\n")
        padded_array = None

    elif args.padding == "direct":
        print("\nWARNING: All sequences have different lengths, adding padding to the sequences with start and end indices using the input_list")
        num_seqs = len(final_array)
        num_features = final_array[0].shape[1]
        if args.add_padding_indicator:
            num_features += 1

        missing_residues = input_list["Missing"].values
        min_start = input_list["Start"].min()
        max_end = input_list["End"].max()

        max_length = max_end - min_start + 1

        padded_array = np.zeros((num_seqs, max_length, num_features), dtype=np.float32)
        print(f"Create padded array with zero element of shape {padded_array.shape}\n")

        for i, seq in enumerate(final_array):
            input_index = int(seq[-1,-1])
            start_index = int(input_list.loc[input_index, "Start"] - min_start)
            end_index = int(input_list.loc[input_index, "End"] - min_start +1 )
            if not pd.isna(missing_residues[input_index]):
                missing_residue = missing_residues[input_index]
                pairs = re.findall(r'\[(\d+),(\d+)\]', missing_residue)

                if not pairs:
                    # Handle original format: "21,34"
                    a, b = map(int, missing_residue.split(','))
                    pairs = [(a, b)]

                # Convert all pairs to array indices
                missing_ranges = [(int(a) - min_start - 1, int(b) - min_start - 1) for a, b in pairs]

            else:
                missing_residue = 0
                missing_ranges = []

            if args.add_padding_indicator:
                padded_array[i, start_index:end_index, :-2] = seq[:, :-1]
                padded_array[i, start_index:end_index, -1] = seq[:, -1]
                padded_array[i, start_index:end_index, -2] = 1

                # Zero out all missing ranges
                for start_missing, end_missing in missing_ranges:
                    padded_array[i, start_missing:end_missing, :] = 0

            else:
                padded_array[i, start_index:end_index, :-1] = seq[:, :-1]
                padded_array[i, start_index:end_index, -1] = seq[:, -1]

                # Zero out all missing ranges
                for start_missing, end_missing in missing_ranges:
                    padded_array[i, start_missing:end_missing, :] = 0
            final_array[i] = None
        
    else:
        ## Get the maximum length of the sequences with start and end indices using the input_list
        print("\nWARNING: All sequences have different lengths.")
        max_length = max(sequence_lengths)
        num_seqs = len(final_array)
        num_features = final_array[0].shape[1]

        if args.add_padding_indicator:
            num_features += 1

        print(f"Sequences have different lengths. Padding with zeros to the maximum length {max_length} with dtype {final_array[0].dtype}.\n")
        padded_array = np.zeros((num_seqs, max_length, num_features), dtype=np.float32)
        print(f"Create padded array with zero element of shape {padded_array.shape}\n")

        for i, seq in enumerate(final_array):
            input_index = int(seq[-1,-1])
            # Add another column to indicate if the coordinates are padded or not (1 for padded, 0 for not padded)
            if args.add_padding_indicator:
                padded_array[i, :seq.shape[0], :-2] = seq[:, :-1]
                padded_array[i, seq.shape[0]:, -2] = 1
                padded_array[i, :seq.shape[0], -1] = seq[:, -1]
            else:
                padded_array[i, :seq.shape[0], :] = seq
            final_array[i] = None

if padded_array is not None:
    print(f"\nSaving final_array to {output_dir}/{file_name}.pkl with shape {np.array(padded_array).shape}...")
    if args.add_padding_indicator:
        print("Added padding indicator column (1 for padded, 0 for not padded) to the last dimension of the array.")
        with open(f"{output_dir}/{file_name}_padded_with_indicator.pkl", "wb") as f:
            pickle.dump(padded_array, f)
    else:
        with open(f"{output_dir}/{file_name}.pkl", "wb") as f:
            pickle.dump(padded_array, f)

elif padded_array is None:
    print(f"\nSaving final_array to {output_dir}/{file_name}.pkl with shape {np.array(final_array).shape}...")
    with open(f"{output_dir}/{file_name}.pkl", "wb") as f:
        pickle.dump(final_array, f)
