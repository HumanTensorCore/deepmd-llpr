import argparse
import numpy as np
from pathlib import Path
import torch
import deepmd.pt
import warnings

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute LLPR scores from DeePMD "
            "last-layer features."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained PyTorch model.",
    )

    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help=(
            "Feature file generated from the structures "
            "used to train the reference model."
        ),
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help=(
            "Trajectory file containing candidate "
            "configurations to evaluate."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dp-llpr-scores.npz"),
        help="Path for the output file.",
    )

    parser.add_argument(
        "--regularizer",
        type=float,
        default=1e-8,
        help="Regularization strength. Default: 1e-8.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Keep the top-k least rigid candidate "
            "configurations."
        ),
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
        "--max-rigidity",
        type=float,
        default=None,
        help=(
            "Select candidates with rigidity "
            "below this value."
        ),
    )

    parser.add_argument(
        "--min-frame-gap",
        type=int,
        default=10,
        help=(
            "Minimum frame gap between selected "
            "configurations. Default: 10."
        ),
    )

    return parser

def validate_args(args, parser):
    if not args.reference.is_file():
        parser.error(
            f"--reference file not exists: {args.reference}."
        )
    if not args.candidates.is_file():
        parser.error(
            f"candidate trajectory file does not exist: "
            f"{args.candidates}"
        )

    if args.regularizer <= 0:
        parser.error(
            "--regularizer should > 0."
        )
    if args.top_k is not None and args.top_k <= 0:
        parser.error(
            "--top-k must be greater than 0."
        )

    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}"
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be greater than 0."
        )

    if (
            args.max_rigidity is not None
            and args.max_rigidity <= 0
    ):
        parser.error(
            "--max-rigidity must be greater than 0."
        )

    if args.min_frame_gap < 0:
        parser.error(
            "--min-frame-gap must be non-negative."
        )

def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
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

def process_candidate_batch(
    model,
    frame_batch,
    device,
    ntypes,
):
    coords = np.stack(
        [
            frame["coords"]
            for frame in frame_batch
        ],
        axis=0,
    )

    boxes = np.stack(
        [
            frame["box"]
            for frame in frame_batch
        ],
        axis=0,
    )

    atom_types = frame_batch[0]["atom_types"]

    batch_size = coords.shape[0]
    natoms = coords.shape[1]

    for frame in frame_batch:
        if frame["coords"].shape != (natoms, 3):
            raise ValueError(
                "all candidate frames must have "
                "the same number of atoms"
            )

        if not np.array_equal(
            frame["atom_types"],
            atom_types,
        ):
            raise ValueError(
                "atom types must be identical "
                "for all frames"
            )

    coord_tensor = torch.tensor(
        coords,
        dtype=torch.float64,
        device=device,
    )

    atype_tensor = torch.tensor(
        atom_types,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(
        batch_size,
        -1,
    ).contiguous()

    box_tensor = torch.tensor(
        boxes,
        dtype=torch.float64,
        device=device,
    )

    model.set_eval_fitting_last_layer_hook(True)

    try:
        with torch.no_grad():
            model(
                coord_tensor,
                atype_tensor,
                box_tensor,
            )

            feature_tensor = (
                model.eval_fitting_last_layer()
            )

    finally:
        model.set_eval_fitting_last_layer_hook(False)

    feature_batch = (
        feature_tensor.detach()
        .cpu()
        .numpy()
    )

    hidden_dim = feature_batch.shape[-1]

    design_blocks = []

    for type_index in range(ntypes):
        mask = atom_types == type_index

        feature_sum = (
            feature_batch[:, mask, :]
            .sum(axis=1)
        )

        type_count = np.full(
            (batch_size, 1),
            mask.sum(),
            dtype=feature_batch.dtype,
        )

        type_block = np.concatenate(
            [
                feature_sum,
                type_count,
            ],
            axis=1,
        )

        design_blocks.append(type_block)

    return np.concatenate(
        design_blocks,
        axis=1,
    )
#处理单帧dump，我要看一下
def extract_candidate_features(
    model,
    candidate_frames,
    device,
    batch_size,
):
    candidate_timesteps = []
    ntypes = model.get_ntypes()
    feature_batches = []

    frame_batch = []

    for frame in candidate_frames:
        candidate_timesteps.append(
            frame["timestep"]
        )
        frame_batch.append(frame)

        if len(frame_batch) < batch_size:
            continue

        batch_features = process_candidate_batch(
            model,
            frame_batch,
            device,
            ntypes,
        )

        feature_batches.append(batch_features)
        frame_batch.clear()

    if frame_batch:
        batch_features = process_candidate_batch(
            model,
            frame_batch,
            device,
            ntypes,
        )

        feature_batches.append(batch_features)

    if not feature_batches:
        raise ValueError(
            "candidate trajectory contains no frames"
        )

    candidate_matrix = np.concatenate(
        feature_batches,
        axis=0,
    )

    return (
        candidate_matrix,
        np.asarray(
            candidate_timesteps,
            dtype=np.int64,
        ),
    )
#处理dump，我要看一下
def read_one_frame(file):
    """Read one frame from a LAMMPS custom dump file."""

    line = file.readline()

    while line and not line.strip():
        line = file.readline()

    if not line:
        return None

    if line.strip() != "ITEM: TIMESTEP":
        raise ValueError(
            "expected 'ITEM: TIMESTEP', "
            f"got: {line.strip()}"
        )

    timestep_line = file.readline()
    timestep = int(timestep_line.strip())

    line = file.readline()
    if line.strip() != "ITEM: NUMBER OF ATOMS":
        raise ValueError(
            "expected 'ITEM: NUMBER OF ATOMS'"
        )

    natoms_line = file.readline()
    natoms = int(natoms_line.strip())

    box_header = file.readline().strip()

    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(
            "expected 'ITEM: BOX BOUNDS'"
        )

    box_rows = []

    for _ in range(3):
        values = file.readline().split()

        if len(values) < 2:
            raise ValueError(
                "invalid box bounds line"
            )

        box_rows.append(
            [float(value) for value in values]
        )

    atoms_header = file.readline().strip()

    if not atoms_header.startswith("ITEM: ATOMS"):
        raise ValueError(
            "expected 'ITEM: ATOMS'"
        )

    columns = atoms_header.split()[2:]

    required_columns = {
        "id",
        "type",
        "x",
        "y",
        "z",
    }

    if not required_columns.issubset(columns):
        raise ValueError(
            "trajectory must contain "
            "id type x y z"
        )

    id_index = columns.index("id")
    type_index = columns.index("type")
    x_index = columns.index("x")
    y_index = columns.index("y")
    z_index = columns.index("z")

    atom_rows = []

    for _ in range(natoms):
        values = file.readline().split()

        if len(values) < len(columns):
            raise ValueError(
                "invalid atom line in trajectory"
            )

        atom_id = int(values[id_index])
        atom_type = int(values[type_index])

        x = float(values[x_index])
        y = float(values[y_index])
        z = float(values[z_index])

        atom_rows.append(
            (atom_id, atom_type, x, y, z)
        )

    atom_rows.sort(key=lambda row: row[0])

    atom_ids = np.asarray(
        [row[0] for row in atom_rows],
        dtype=np.int64,
    )

    atom_types = np.asarray(
        [row[1] - 1 for row in atom_rows],
        dtype=np.int64,
    )

    coords = np.asarray(
        [
            [row[2], row[3], row[4]]
            for row in atom_rows
        ],
        dtype=np.float64,
    )

    if len(np.unique(atom_ids)) != natoms:
        raise ValueError(
            "atom IDs are not unique"
        )

    # Restricted triclinic LAMMPS box:
    # xlo_bound xhi_bound xy
    # ylo_bound yhi_bound xz
    # zlo_bound zhi_bound yz

    xlo_bound, xhi_bound, xy = box_rows[0]
    ylo_bound, yhi_bound, xz = box_rows[1]
    zlo_bound, zhi_bound, yz = box_rows[2]

    xlo = xlo_bound - min(
        0.0,
        xy,
        xz,
        xy + xz,
    )

    ylo = ylo_bound - min(
        0.0,
        yz,
    )

    xhi = xhi_bound - max(
        0.0,
        xy,
        xz,
        xy + xz,
    )

    yhi = yhi_bound - max(
        0.0,
        yz,
    )

    zlo = zlo_bound
    zhi = zhi_bound

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    box = np.asarray(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=np.float64,
    )

    return {
        "timestep": timestep,
        "coords": coords,
        "atom_types": atom_types,
        "box": box,
    }

def iter_lammps_frames(trajectory_path: Path):
    with trajectory_path.open("r") as file:
        while True:
            frame = read_one_frame(file)

            if frame is None:
                break

            yield frame


def load_feature_file(path: Path, name: str):
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(
            f"cannot load {name} feature file "
            f"{path}: {error}"
        )

    if "design_matrix" not in data.files:
        raise ValueError(
            f"{name} file does not contain 'design_matrix': "
            f"{path}. Please provide an NPZ file generated by "
            "extract_dp_features.py."
        )

    design_matrix = np.asarray(
        data["design_matrix"],
        dtype=np.float64,
    )

    if design_matrix.ndim != 2:
        raise ValueError(
            f"{name} design matrix must be 2D, "
            f"got shape {design_matrix.shape}."
        )

    if design_matrix.shape[0] == 0:
        raise ValueError(
            f"{name} design matrix is empty."
        )

    if not np.all(np.isfinite(design_matrix)):
        raise ValueError(
            f"{name} design matrix contains "
            "NaN or infinite values."
        )

    if "system_indices" not in data.files:
        raise ValueError(
            "reference file does not contain "
            "'system_indices'."
        )

    if "frame_indices" not in data.files:
        raise ValueError(
            "reference file does not contain "
            "'frame_indices'."
        )

    system_indices = np.asarray(
        data["system_indices"],
        dtype=np.int64,
    )

    frame_indices = np.asarray(
        data["frame_indices"],
        dtype=np.int64,
    )

    return (
        design_matrix,
        system_indices,
        frame_indices,
    )

def resolve_device(device):
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    elif (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        warnings.warn(
            "CUDA was requested but is not available; "
            "falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        device = "cpu"

    return torch.device(device)

#计算llpr，我要看一下
def compute_llpr_scores(
    reference_matrix,
    candidate_matrix,
    regularizer,
):
    feature_dim = reference_matrix.shape[1]

    covariance = (
        reference_matrix.T @ reference_matrix
    )

    covariance += (
        regularizer
        * np.eye(
            feature_dim,
            dtype=reference_matrix.dtype,
        )
    )

    solved = np.linalg.solve(
        covariance,
        candidate_matrix.T,
    ).T

    leverage = np.sum(
        candidate_matrix * solved,
        axis=1,
    )

    if not np.all(np.isfinite(leverage)):
        raise ValueError(
            "LLPR leverage contains NaN or infinite values."
        )

    tolerance = 1e-12

    if np.any(leverage < -tolerance):
        raise ValueError(
            "LLPR leverage contains significantly negative values."
        )

    leverage = np.maximum(leverage, 0.0)
    rigidity = np.full_like(leverage, np.inf)
    positive = leverage > tolerance
    rigidity[positive] = 1.0 / leverage[positive]

    return rigidity

#归一化，我要看一下
def normalize_features(
    reference_matrix,
    candidate_matrix,
):
    feature_mean = reference_matrix.mean(
        axis=0,
    )

    feature_std = reference_matrix.std(
        axis=0,
    )

    feature_std[feature_std < 1e-12] = 1.0

    reference_scaled = (
        reference_matrix - feature_mean
    ) / feature_std

    candidate_scaled = (
        candidate_matrix - feature_mean
    ) / feature_std

    return (
        reference_scaled,
        candidate_scaled,
    )
#筛选函数，我需要看一下
def select_candidates(
    rigidity,
    max_rigidity=None,
    top_k=None,
    min_frame_gap=0,
):
    if max_rigidity is None:
        eligible = np.arange(
            rigidity.size,
            dtype=np.int64,
        )
    else:
        eligible = np.flatnonzero(
            rigidity <= max_rigidity
        )

    eligible = eligible[
        np.argsort(rigidity[eligible])
    ]

    selected = []

    for index in eligible:
        if any(
            abs(index - old_index)
            < min_frame_gap
            for old_index in selected
        ):
            continue

        selected.append(int(index))

        if (
            top_k is not None
            and len(selected) >= top_k
        ):
            break

    return np.asarray(
        selected,
        dtype=np.int64,
    )

def main():
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args, parser)

    (
        reference_matrix,
        system_indices,
        frame_indices,
    ) = load_feature_file(
        args.reference,
        "reference",
    )

    device = resolve_device(args.device)

    model = load_model(
        args.model,
        device,
    )

    candidate_frames = iter_lammps_frames(
        args.candidates,
    )

    (
        candidate_matrix,
        candidate_timesteps,
    ) = extract_candidate_features(
        model,
        candidate_frames,
        device,
        args.batch_size,
    )

    (
        reference_scaled,
        candidate_scaled,
    ) = normalize_features(
        reference_matrix,
        candidate_matrix,
    )

    reference_rigidity = compute_llpr_scores(
        reference_scaled,
        reference_scaled,
        args.regularizer,
    )

    candidate_rigidity = compute_llpr_scores(
        reference_scaled,
        candidate_scaled,
        args.regularizer,
    )

    print(
        "normalized reference minimum:",
        reference_rigidity.min(),
    )

    print(
        "normalized reference maximum:",
        reference_rigidity.max(),
    )

    print(
        "normalized reference mean:",
        reference_rigidity.mean(),
    )

    print(
        "normalized candidate minimum:",
        candidate_rigidity.min(),
    )

    print(
        "normalized candidate maximum:",
        candidate_rigidity.max(),
    )

    print(
        "normalized candidate mean:",
        candidate_rigidity.mean(),
    )

    percentiles = [1, 5, 25, 50, 75, 95, 99]

    reference_quantiles = np.percentile(
        reference_rigidity,
        percentiles,
    )

    candidate_quantiles = np.percentile(
        candidate_rigidity,
        percentiles,
    )

    print(
        "percentile | reference | candidate"
    )

    for (
        percentile,
        reference_value,
        candidate_value,
    ) in zip(
        percentiles,
        reference_quantiles,
        candidate_quantiles,
    ):
        print(
            f"{percentile:9d}% | "
            f"{reference_value:9.4f} | "
            f"{candidate_value:9.4f}"
        )

    selected_indices = select_candidates(
        candidate_rigidity,
        max_rigidity=args.max_rigidity,
        top_k=args.top_k,
        min_frame_gap=args.min_frame_gap,
    )

    print(
        "selected candidates:",
        selected_indices.size,
    )

    for index in selected_indices:
        print(
            "candidate index:",
            index,
            "timestep:",
            candidate_timesteps[index],
            "rigidity:",
            candidate_rigidity[index],
        )
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        args.output,
        candidate_indices=selected_indices,
        candidate_timesteps=candidate_timesteps[
            selected_indices
        ],
        candidate_rigidity=candidate_rigidity[
            selected_indices
        ],
        all_candidate_rigidity=candidate_rigidity,
        reference_rigidity=reference_rigidity,
    )

    print("saved:", args.output)

if __name__ == "__main__":
    main()

