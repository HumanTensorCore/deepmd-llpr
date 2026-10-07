import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
import deepmd.pt

def build_parser():
    """Define command-line arguments for reference feature extraction."""
    parser = argparse.ArgumentParser(
        description=("Extract last-layer atomic features "
                     "from a trained DeePMD PyTorch model.")
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained PyTorch model.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="Path to the dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dp_features.npz"),
        help="Path to the output file.",
    )

    parser.add_argument(
        "--frames",
        type=str,
        default="all",
        help=("frames to process: 'all', one index such as '10',"
              "or a range such as '100:500' or '0:1000:10'."
              "Default: all"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=("Batch size. Default: 32"),
    )

    parser.add_argument(
        "--device",
        choices=["auto","cpu", "cuda"],
        default="auto",
        help=("Device used for training. Default: auto"),
    )

    parser.add_argument(
        "--set",
        dest="set_name",
        default="set.000",
        help="DeepMD set directory name. Default: set.000.",
    )
    return parser

def validate_args(args, parser):
    """Validate datasets, model path, set names, and batch size."""
    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}"
        )

    set_dirs = []

    for dataset in args.dataset:
        if not dataset.is_dir():
            parser.error(
                f"dataset directory does not exist: {dataset}"
            )

        set_dir = dataset / args.set_name

        if not set_dir.is_dir():
            parser.error(f"set directory does not exist: {set_dir}"
                         )
        set_dirs.append(set_dir)

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    return set_dirs

def prepare_output_path(
    output_path: Path,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate the output path and create its parent directories."""

    if output_path.exists() and output_path.is_dir():
        parser.error(
            f"output path is a directory, not a file: {output_path}"
        )

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as error:
        parser.error(
            f"cannot create output directory: {error}"
        )
def resolve_device(device):
    """Resolve auto/CPU/CUDA selection to a torch.device."""
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    elif device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA was requested but is not available; "
            "falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        device = "cpu"
    return torch.device(device)

def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a TorchScript model and verify its last-layer hook methods."""
    print("loading model...", model_path)

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    required_methods = (
        "set_eval_fitting_last_layer_hook",
        "eval_fitting_last_layer",
    )
    for method_name in required_methods:
        if not hasattr(model, method_name):
            raise RuntimeError(
                "Model does not provide the required method: "
                f"{method_name}"
            )
    return model

def load_set_data(set_dir: Path):
    """Load coordinates, boxes, atom types, and type names from one set."""
    coord_path = set_dir / "coord.npy"
    box_path = set_dir / "box.npy"
    type_path = set_dir.parent / "type.raw"
    type_map_path = set_dir.parent / "type_map.raw"

    if not coord_path.is_file():
        raise FileNotFoundError(
            f"coordinate file does not exist: {coord_path}"
        )

    coords = np.load(coord_path,mmap_mode="r")
    if box_path.is_file():
        boxes = np.load(box_path,mmap_mode="r")
    else:
        boxes = None


    if not type_path.is_file():
        raise FileNotFoundError(
            f"atom type file does not exist: {type_path}"
        )

    atom_types = np.loadtxt(type_path,dtype=np.int64,)

    if not type_map_path.is_file():
        raise FileNotFoundError(
            f"type map file does not exist: {type_map_path}"
        )

    type_map = type_map_path.read_text().split()

    return coords, boxes, atom_types, type_map

def parse_frames(
    frame_spec: str,
    nframes: int,
) -> np.ndarray:
    """Convert a frame expression into validated integer frame indices."""
    frame_spec = frame_spec.strip()

    if frame_spec == "all":
        return np.arange(
            nframes,
            dtype=np.int64,
        )

    if ":" in frame_spec:
        parts = frame_spec.split(":")

        if len(parts) > 3:
            raise ValueError(
                f"invalid frame range: {frame_spec}"
            )

        values = [
            int(part) if part else None
            for part in parts
        ]

        frame_slice = slice(*values)

        try:
            start, stop, step = frame_slice.indices(
                nframes
            )
        except ValueError as error:
            raise ValueError(
                f"invalid frame range: {frame_spec}"
            ) from error

        indices = np.arange(
            start,
            stop,
            step,
            dtype=np.int64,
        )

    else:
        try:
            index = int(frame_spec)
        except ValueError as error:
            raise ValueError(
                f"invalid frame index: {frame_spec}"
            ) from error

        if index < 0:
            index += nframes

        if index < 0 or index >= nframes:
            raise ValueError(
                f"frame index {frame_spec} is out of range "
                f"for a set containing {nframes} frames"
            )

        indices = np.array(
            [index],
            dtype=np.int64,
        )

    if indices.size == 0:
        raise ValueError(
            f"frame selection is empty: {frame_spec}"
        )

    return indices

def main():
    """Extract reference matrices and save them as a compressed NPZ file."""
    parser = build_parser()
    args = parser.parse_args()
    set_dirs = validate_args(args, parser)
    prepare_output_path(args.output, parser)
    device = resolve_device(args.device)
    model = load_model(args.model, device)

    print("resolved device:", device)
    print("output:", args.output)
    print("frames:", args.frames)
    print("batch size:", args.batch_size)

    print("datasets:")
    for dataset in args.dataset:
        print("  ", dataset)

    print("set directories:")
    for set_dir in set_dirs:
        print("  ", set_dir)

    print("resolved device:", device)
    print("type map:", model.get_type_map())
    print("number of types:", model.get_ntypes())
    print("cutoff:", model.get_rcut())
    print("fparam dimension:", model.get_dim_fparam())
    print("aparam dimension:", model.get_dim_aparam())

    design_matrix_batches = []
    system_index_batches = []
    frame_index_batches = []
    set_paths = []

    for system_index, set_dir in enumerate(set_dirs):
        # ---------- Current set ----------
        coords, boxes, atom_types, type_map = (
            load_set_data(set_dir)
        )

        print("coords:", coords.shape)
        print(
            "boxes:",
            None if boxes is None else boxes.shape,
        )
        print("atom types:", atom_types.shape)
        print("type map:", type_map)

        # Check that the dataset and model use the same type map
        model_type_map = list(model.get_type_map())

        if type_map != model_type_map:
            raise ValueError(
                f"type map mismatch: dataset={type_map}, "
                f"model={model_type_map}"
            )

        nframes = coords.shape[0]

        try:
            frame_indices = parse_frames(
                args.frames,
                nframes,
            )
        except ValueError as error:
            parser.error(f"{set_dir}: {error}")

        print("set:", set_dir)
        print("total frames:", nframes)
        print("selected frames:", frame_indices.size)
        print("first selected index:", frame_indices[0])
        print("last selected index:", frame_indices[-1])

        # Store results from all batches in the current set
        type_feature_batches = []
        type_count_batches = []

        for start in range(
                0,
                len(frame_indices),
                args.batch_size,
        ):
            # ---------- Current batch ----------
            stop = start + args.batch_size
            batch_indices = frame_indices[start:stop]

            coord_batch = np.asarray(
                coords[batch_indices]
            )

            if boxes is None:
                box_batch = None
            else:
                box_batch = np.asarray(
                    boxes[batch_indices]
                )

            print("frame indices:", batch_indices)
            print("coord batch shape:", coord_batch.shape)
            print(
                "box batch shape:",
                None if box_batch is None
                else box_batch.shape,
            )

            if coord_batch.shape[1] % 3 != 0:
                raise ValueError(
                    f"invalid coordinate shape: "
                    f"{coord_batch.shape}"
                )

            current_batch_size = coord_batch.shape[0]
            natoms = coord_batch.shape[1] // 3

            if atom_types.shape != (natoms,):
                raise ValueError(
                    f"atom types shape {atom_types.shape} "
                    f"does not match {natoms} atoms"
                )

            ntypes = model.get_ntypes()

            if np.any(atom_types < 0) or np.any(
                atom_types >= ntypes
            ):
                raise ValueError(
                    f"atom types must be in the range "
                    f"[0, {ntypes - 1}]"
                )

            # ---------- Convert NumPy arrays to PyTorch tensors ----------
            coord_tensor = torch.tensor(
                coord_batch,
                device=device,
            ).reshape(
                current_batch_size,
                natoms,
                3,
            )

            atype_tensor = torch.tensor(
                atom_types,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(
                current_batch_size,
                -1,
            ).contiguous()

            if box_batch is None:
                box_tensor = None
            else:
                box_tensor = torch.tensor(
                    box_batch,
                    device=device,
                ).reshape(
                    current_batch_size,
                    3,
                    3,
                )

            # Expose the DeePMD fitting network's last-layer atomic
            # representation during the forward pass.
            model.set_eval_fitting_last_layer_hook(
                True
            )

            try:
                _ = model(
                    coord_tensor,
                    atype_tensor,
                    box_tensor,
                )

                feature_tensor = (
                    model.eval_fitting_last_layer()
                )
            finally:
                model.set_eval_fitting_last_layer_hook(
                    False
                )

            feature_batch = (
                feature_tensor
                .detach()
                .cpu()
                .numpy()
            )

            hidden_dim = feature_batch.shape[-1]

            print("feature shape:", feature_batch.shape)
            print("feature dtype:", feature_batch.dtype)

            if feature_batch.shape[:2] != (
                    current_batch_size,
                    natoms,
            ):
                raise ValueError(
                    f"unexpected feature shape: "
                    f"{feature_batch.shape}"
                )

            # ---------- Sum atomic features by atom type ----------
            ntypes = model.get_ntypes()
            hidden_dim = feature_batch.shape[-1]

            # Aggregate per-atom representations by chemical type so that
            # every configuration becomes one fixed-length feature vector.
            type_features = np.zeros(
                (
                    current_batch_size,
                    ntypes,
                    hidden_dim,
                ),
                dtype=feature_batch.dtype,
            )

            type_counts = np.zeros(
                (
                    current_batch_size,
                    ntypes,
                ),
                dtype=feature_batch.dtype,
            )

            for type_index in range(ntypes):
                mask = atom_types == type_index

                type_features[:, type_index, :] = (
                    feature_batch[:, mask, :].sum(
                        axis=1
                    )
                )

                type_counts[:, type_index] = (
                    mask.sum()
                )

            # Store the current batch results
            type_feature_batches.append(
                type_features
            )
            type_count_batches.append(
                type_counts
            )

        # ---------- End of the batch loop ----------

        # Concatenate all batches from the current set
        set_type_features = np.concatenate(
            type_feature_batches,
            axis=0,
        )

        set_type_counts = np.concatenate(
            type_count_batches,
            axis=0,
        )

        print(
            "all type features:",
            set_type_features.shape,
        )
        print(
            "all type counts:",
            set_type_counts.shape,
        )

        # Build the configuration-level LLPR design matrix. Each type
        # contributes summed hidden features and its atom count.
        design_blocks = []

        for type_index in range(ntypes):
            type_block = np.concatenate(
                [
                    set_type_features[
                        :,
                        type_index,
                        :,
                    ],
                    set_type_counts[
                        :,
                        type_index:type_index + 1,
                    ],
                ],
                axis=1,
            )

            design_blocks.append(type_block)

        set_design_matrix = np.concatenate(
            design_blocks,
            axis=1,
        )

        print(
            "design matrix:",
            set_design_matrix.shape,
        )

        design_matrix_batches.append(
            set_design_matrix
        )

        system_index_batches.append(
            np.full(
                frame_indices.shape,
                system_index,
                dtype=np.int64,
            )
        )

        frame_index_batches.append(
            frame_indices.copy()
        )

        set_paths.append(str(set_dir))

    # ---------- End of the set loop ----------

    # Concatenate results from all sets
    design_matrix = np.concatenate(
        design_matrix_batches,
        axis=0,
    )

    system_indices = np.concatenate(
        system_index_batches,
        axis=0,
    )

    selected_frame_indices = np.concatenate(
        frame_index_batches,
        axis=0,
    )

    print(
        "final design matrix:",
        design_matrix.shape,
    )
    print(
        "system indices:",
        system_indices.shape,
    )
    print(
        "frame indices:",
        selected_frame_indices.shape,
    )

    # Save the extracted LLPR features
    np.savez_compressed(
        args.output,
        design_matrix=design_matrix,
        system_indices=system_indices,
        frame_indices=selected_frame_indices,
        type_map=np.asarray(
            model.get_type_map()
        ),
        set_paths=np.asarray(set_paths),
        hidden_dim=np.asarray(hidden_dim),
        include_bias=np.asarray(True),
    )

    print("saved:", args.output)


    # parser.print_help()
    # print(args)
if __name__ == "__main__":
    main()

