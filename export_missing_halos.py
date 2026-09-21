#!/usr/bin/env python3
"""Export bounded-memory, galaxy-centered regions from TNG50-1 parent halos."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import shutil
import sys
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError:  # Allow --list-targets to work outside the TNG Lab.
    h5py = None
    np = None


DEFAULT_SIMULATION = "TNG50-1"
DEFAULT_OUTPUT_DIR = Path("snap99_galaxy_cutouts_300kpc")
DEFAULT_SIZE_KPC = 300.0
DEFAULT_MEMORY_BUDGET_MIB = 256
DEFAULT_MIN_FREE_GIB = 5.0
TARGETS_FILE = Path(__file__).with_name("targets_snap99.csv")

PARTICLE_FIELDS = {
    0: (
        "Coordinates",
        "Masses",
        "GFM_Metallicity",
        "Density",
        "ElectronAbundance",
        "InternalEnergy",
        "StarFormationRate",
    ),
    4: (
        "Coordinates",
        "GFM_StellarFormationTime",
        "GFM_InitialMass",
        "GFM_Metallicity",
    ),
}


@dataclass(frozen=True, order=True)
class Target:
    snapshot: int
    subhalo: int
    parent_halo: int

    @property
    def key(self) -> str:
        return f"{self.snapshot}:{self.subhalo}"


@dataclass(frozen=True)
class HaloFileRange:
    path: Path
    source_start: int
    source_stop: int
    halo_start: int

    @property
    def count(self) -> int:
        return self.source_stop - self.source_start


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_targets(path: Path) -> list[Target]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        targets = [
            Target(
                snapshot=int(row["snapshot"]),
                subhalo=int(row["subhalo"]),
                parent_halo=int(row["parent_halo"]),
            )
            for row in rows
        ]
    keys = [(target.snapshot, target.subhalo) for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate snapshot/subhalo entries in {path}")
    return targets


def locate_base_path(simulation: str, explicit: str | None) -> Path:
    candidates: Iterable[Path]
    if explicit:
        candidates = (Path(explicit).expanduser(),)
    else:
        candidates = (
            Path("sims.TNG") / simulation / "output",
            Path.home() / "sims.TNG" / simulation / "output",
            Path("/home/jovyan/sims.TNG") / simulation / "output",
        )

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the mounted simulation output. Checked:\n  "
        + "\n  ".join(checked)
        + "\nPass it explicitly with --base-path."
    )


def snapshot_piece(base_path: Path, snapshot: int) -> Path:
    path = base_path / f"snapdir_{snapshot:03d}" / f"snap_{snapshot:03d}.0.hdf5"
    if not path.is_file():
        alternate = path.with_name(f"snapshot_{snapshot:03d}.0.hdf5")
        if alternate.is_file():
            return alternate
        raise FileNotFoundError(f"Snapshot header file not found: {path}")
    return path


def copy_snapshot_metadata(source_path: Path, destination: h5py.File) -> None:
    with h5py.File(source_path, "r", rdcc_nbytes=1024 * 1024) as source:
        for group_name in ("Header", "Parameters", "Configuration"):
            if group_name not in source:
                continue
            target_group = destination.require_group(group_name)
            for key, value in source[group_name].attrs.items():
                target_group.attrs[key] = value


def snapshot_geometry(source_path: Path) -> tuple[float, float, float]:
    with h5py.File(source_path, "r", rdcc_nbytes=1024 * 1024) as source:
        header = source["Header"].attrs
        return (
            float(header["Time"]),
            float(header["HubbleParam"]),
            float(header["BoxSize"]),
        )


def halo_file_ranges(il, base_path: Path, target: Target, particle_type: int):
    """Resolve one FoF halo into source-file slices without loading its particles."""
    subset = il.snapshot.getSnapOffsets(
        str(base_path), target.snapshot, target.parent_halo, "Group"
    )
    total = int(subset["lenType"][particle_type])
    if total == 0:
        return 0, []

    snap_offsets = np.asarray(subset["snapOffsets"])
    if snap_offsets.ndim != 2:
        raise RuntimeError(f"Unexpected snapOffsets shape: {snap_offsets.shape}")
    if snap_offsets.shape[0] != 6:
        if snap_offsets.shape[1] == 6:
            snap_offsets = snap_offsets.T
        else:
            raise RuntimeError(f"Unexpected snapOffsets shape: {snap_offsets.shape}")

    global_start = int(subset["offsetType"][particle_type])
    offsets_this_type = global_start - snap_offsets[particle_type, :]
    candidates = np.where(offsets_this_type >= 0)[0]
    if not len(candidates):
        raise RuntimeError("Could not locate the halo's first snapshot file")

    file_number = int(candidates[-1])
    file_offset = int(offsets_this_type[file_number])
    remaining = total
    halo_cursor = 0
    ranges = []

    while remaining:
        if file_number >= snap_offsets.shape[1]:
            raise RuntimeError(
                f"Snapshot pieces ended with {remaining} PartType{particle_type} "
                "particles still unread"
            )
        path = Path(il.snapshot.snapPath(str(base_path), target.snapshot, file_number))
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot piece not found: {path}")
        with h5py.File(path, "r", rdcc_nbytes=1024 * 1024) as source:
            local_count = int(source["Header"].attrs["NumPart_ThisFile"][particle_type])

        available = max(0, local_count - file_offset)
        count = min(remaining, available)
        if count:
            ranges.append(
                HaloFileRange(
                    path=path,
                    source_start=file_offset,
                    source_stop=file_offset + count,
                    halo_start=halo_cursor,
                )
            )
            remaining -= count
            halo_cursor += count
        file_number += 1
        file_offset = 0

    if halo_cursor != total:
        raise RuntimeError(f"Resolved {halo_cursor} particles, expected {total}")
    return total, ranges


def field_spec(ranges: list[HaloFileRange], particle_type: int, field: str):
    group_name = f"PartType{particle_type}"
    for file_range in ranges:
        with h5py.File(file_range.path, "r", rdcc_nbytes=1024 * 1024) as source:
            if group_name not in source or field not in source[group_name]:
                continue
            dataset = source[group_name][field]
            return np.dtype(dataset.dtype), tuple(dataset.shape[1:])
    raise KeyError(f"Could not find {group_name}/{field} in the halo's snapshot pieces")


def bytes_per_row(dtype, tail_shape: tuple[int, ...]) -> int:
    elements = int(np.prod(tail_shape, dtype=np.int64)) if tail_shape else 1
    return int(np.dtype(dtype).itemsize * elements)


def safe_chunk_rows(memory_budget_bytes: int, row_bytes: int) -> int:
    # Source buffer, selected copy, arithmetic workspace, and a conservative margin.
    per_row_budget = max(64, row_bytes * 4)
    return max(1, min(1_000_000, memory_budget_bytes // per_row_budget))


def periodic_cube_mask(
    coordinates: np.ndarray,
    center: np.ndarray,
    box_size: float,
    half_width_comoving_kpc_over_h: float,
) -> np.ndarray:
    """Select a periodic cube while allocating only one coordinate column at a time."""
    selected = np.ones(len(coordinates), dtype=np.bool_)
    for axis in range(3):
        delta = np.array(coordinates[:, axis], copy=True)
        scalar_type = delta.dtype.type
        delta -= scalar_type(center[axis])
        delta -= scalar_type(box_size) * np.rint(delta / scalar_type(box_size))
        selected &= np.abs(delta) <= scalar_type(half_width_comoving_kpc_over_h)
        del delta
    return selected


def ensure_disk_space(path: Path, needed_bytes: int, min_free_gib: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    reserve = int(min_free_gib * 1024**3)
    if free < needed_bytes + reserve:
        raise RuntimeError(
            f"Insufficient free space at {path}: {free / 1024**3:.2f} GiB free, "
            f"need about {needed_bytes / 1024**3:.2f} GiB plus the "
            f"{min_free_gib:g} GiB safety reserve"
        )


def scratch_path(
    work_dir: Path, target: Target, particle_type: int, label: str
) -> Path:
    safe_label = "".join(char if char.isalnum() else "_" for char in label)
    return work_dir / (
        f"s{target.snapshot:03d}_sub{target.subhalo}_halo{target.parent_halo}_"
        f"pt{particle_type}_{safe_label}.mmap"
    )


def build_selection_mask(
    ranges: list[HaloFileRange],
    total: int,
    particle_type: int,
    target: Target,
    center: np.ndarray,
    box_size: float,
    half_width: float,
    memory_budget_bytes: int,
    work_dir: Path,
    min_free_gib: float,
):
    coordinate_dtype, coordinate_tail = field_spec(ranges, particle_type, "Coordinates")
    if coordinate_tail != (3,):
        raise RuntimeError(f"Unexpected Coordinates shape per row: {coordinate_tail}")
    row_bytes = bytes_per_row(coordinate_dtype, coordinate_tail)
    chunk_rows = safe_chunk_rows(memory_budget_bytes, row_bytes)

    mask_path = scratch_path(work_dir, target, particle_type, "selection")
    buffer_path = scratch_path(work_dir, target, particle_type, "coordinates_buffer")
    ensure_disk_space(work_dir, total + chunk_rows * row_bytes, min_free_gib)
    mask = np.memmap(mask_path, mode="w+", dtype=np.bool_, shape=(total,))
    buffer = np.memmap(
        buffer_path,
        mode="w+",
        dtype=coordinate_dtype,
        shape=(chunk_rows, *coordinate_tail),
    )

    selected_count = 0
    group_name = f"PartType{particle_type}"
    try:
        for file_range in ranges:
            with h5py.File(file_range.path, "r", rdcc_nbytes=1024 * 1024) as source:
                dataset = source[group_name]["Coordinates"]
                for source_start in range(
                    file_range.source_start, file_range.source_stop, chunk_rows
                ):
                    source_stop = min(source_start + chunk_rows, file_range.source_stop)
                    count = source_stop - source_start
                    halo_start = file_range.halo_start + (
                        source_start - file_range.source_start
                    )
                    halo_stop = halo_start + count
                    dataset.read_direct(
                        buffer,
                        source_sel=np.s_[source_start:source_stop, ...],
                        dest_sel=np.s_[:count, ...],
                    )
                    chunk_mask = periodic_cube_mask(
                        buffer[:count], center, box_size, half_width
                    )
                    mask[halo_start:halo_stop] = chunk_mask
                    selected_count += int(np.count_nonzero(chunk_mask))
                    del chunk_mask
                # Push dirty mask pages to disk regularly instead of allowing a
                # very large halo mask to accumulate in the cgroup page cache.
                mask.flush()
        mask.flush()
    finally:
        del buffer
        buffer_path.unlink(missing_ok=True)
        gc.collect()
    return mask, mask_path, selected_count


def write_field_from_mask(
    destination: h5py.File,
    ranges: list[HaloFileRange],
    mask: np.memmap,
    selected_count: int,
    particle_type: int,
    field: str,
    target: Target,
    memory_budget_bytes: int,
    work_dir: Path,
    min_free_gib: float,
) -> None:
    group = destination.require_group(f"PartType{particle_type}")
    dtype, tail_shape = field_spec(ranges, particle_type, field)
    expected_shape = (selected_count, *tail_shape)
    temporary_name = f"__incomplete__{field}"

    if field in group:
        if group[field].shape != expected_shape or group[field].dtype != dtype:
            raise RuntimeError(
                f"Existing partial dataset {group[field].name} is "
                f"{group[field].shape}/{group[field].dtype}, expected "
                f"{expected_shape}/{dtype}"
            )
        return
    if temporary_name in group:
        del group[temporary_name]

    row_bytes = bytes_per_row(dtype, tail_shape)
    estimated_bytes = selected_count * row_bytes
    ensure_disk_space(Path(destination.filename).parent, estimated_bytes, min_free_gib)
    chunk_rows = safe_chunk_rows(memory_budget_bytes, row_bytes)
    hdf5_chunk_rows = max(1, min(selected_count, chunk_rows, 65_536))
    dataset = group.create_dataset(
        temporary_name,
        shape=expected_shape,
        dtype=dtype,
        chunks=(hdf5_chunk_rows, *tail_shape),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )

    buffer_path = scratch_path(work_dir, target, particle_type, f"{field}_buffer")
    ensure_disk_space(work_dir, chunk_rows * row_bytes, min_free_gib)
    buffer = np.memmap(
        buffer_path,
        mode="w+",
        dtype=dtype,
        shape=(chunk_rows, *tail_shape),
    )
    output_cursor = 0
    group_name = f"PartType{particle_type}"
    try:
        for file_range in ranges:
            with h5py.File(file_range.path, "r", rdcc_nbytes=1024 * 1024) as source:
                source_dataset = source[group_name][field]
                for source_start in range(
                    file_range.source_start, file_range.source_stop, chunk_rows
                ):
                    source_stop = min(source_start + chunk_rows, file_range.source_stop)
                    count = source_stop - source_start
                    halo_start = file_range.halo_start + (
                        source_start - file_range.source_start
                    )
                    halo_stop = halo_start + count
                    chunk_mask = np.asarray(mask[halo_start:halo_stop], dtype=np.bool_)
                    selected_here = int(np.count_nonzero(chunk_mask))
                    if not selected_here:
                        del chunk_mask
                        continue
                    source_dataset.read_direct(
                        buffer,
                        source_sel=np.s_[source_start:source_stop, ...],
                        dest_sel=np.s_[:count, ...],
                    )
                    selected_values = np.asarray(buffer[:count][chunk_mask])
                    dataset[output_cursor : output_cursor + selected_here] = (
                        selected_values
                    )
                    output_cursor += selected_here
                    del selected_values, chunk_mask
        if output_cursor != selected_count:
            raise RuntimeError(
                f"Wrote {output_cursor} rows for PartType{particle_type}/{field}, "
                f"expected {selected_count}"
            )
        destination.flush()
        group.move(temporary_name, field)
        destination.flush()
    finally:
        del buffer
        buffer_path.unlink(missing_ok=True)
        gc.collect()


def set_particle_counts(destination: h5py.File, counts: np.ndarray) -> None:
    header = destination.require_group("Header")
    totals = counts.astype(np.uint64)
    header.attrs["NumPart_ThisFile"] = (totals & np.uint64(0xFFFFFFFF)).astype(
        np.uint32
    )
    header.attrs["NumPart_Total"] = (totals & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    header.attrs["NumPart_Total_HighWord"] = (totals >> np.uint64(32)).astype(np.uint32)
    header.attrs["NumFilesPerSnapshot"] = np.int32(1)


def validate_existing_cutout(
    path: Path,
    target: Target,
    size_kpc: float,
    *,
    allow_uninitialized: bool = False,
) -> None:
    with h5py.File(path, "r", rdcc_nbytes=1024 * 1024) as source:
        actual_snapshot = int(source.attrs.get("SnapshotNumber", -1))
        actual_subhalo = int(source.attrs.get("SubhaloID", -1))
        actual_parent = int(source.attrs.get("ParentHaloID", -1))
        actual_size = float(source.attrs.get("CutoutSizePhysicalKpc", -1.0))
        present = set(source.attrs)
    if allow_uninitialized:
        partial_mismatch = (
            ("SnapshotNumber" in present and actual_snapshot != target.snapshot)
            or ("SubhaloID" in present and actual_subhalo != target.subhalo)
            or ("ParentHaloID" in present and actual_parent != target.parent_halo)
            or (
                "CutoutSizePhysicalKpc" in present
                and not np.isclose(actual_size, size_kpc)
            )
        )
        if not partial_mismatch:
            return
    if (
        actual_snapshot != target.snapshot
        or actual_subhalo != target.subhalo
        or actual_parent != target.parent_halo
        or not np.isclose(actual_size, size_kpc)
    ):
        raise RuntimeError(
            f"{path} exists but is not the requested {size_kpc:g} kpc cutout "
            f"for subhalo {target.subhalo}. Move it aside or choose another --output-dir."
        )


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def export_target(
    base_path: Path,
    output_dir: Path,
    simulation: str,
    target: Target,
    size_kpc: float,
    memory_budget_mib: int,
    work_dir: Path,
    min_free_gib: float,
) -> Path:
    if h5py is None or np is None:
        raise RuntimeError(
            "h5py and numpy are required for export. They are preinstalled in "
            "the IllustrisTNG Lab; elsewhere, run: pip install -r requirements.txt"
        )
    try:
        import illustris_python as il
    except ImportError as exc:
        raise RuntimeError(
            "illustris_python is unavailable. Run this in the IllustrisTNG Lab, "
            "or install the official illustris_python package."
        ) from exc

    final_path = output_dir / (
        f"cutout_{simulation}_{target.snapshot}_{target.subhalo}_"
        f"{target.parent_halo}.hdf5"
    )
    partial_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists():
        validate_existing_cutout(final_path, target, size_kpc)
        print(f"SKIP existing {final_path}", flush=True)
        return final_path

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_disk_space(output_dir, 0, min_free_gib)
    source_path = snapshot_piece(base_path, target.snapshot)
    scale_factor, hubble_parameter, box_size = snapshot_geometry(source_path)
    half_width = 0.5 * size_kpc * hubble_parameter / scale_factor
    memory_budget_bytes = memory_budget_mib * 1024**2

    subhalo_record = il.groupcat.loadSingle(
        str(base_path), target.snapshot, subhaloID=target.subhalo
    )
    center = np.asarray(subhalo_record["SubhaloPos"], dtype=np.float64)
    actual_parent = int(subhalo_record["SubhaloGrNr"])
    if actual_parent != target.parent_halo:
        raise ValueError(
            f"Target manifest says parent halo {target.parent_halo}, but the group "
            f"catalog reports {actual_parent} for snapshot {target.snapshot}, "
            f"subhalo {target.subhalo}"
        )

    if partial_path.exists():
        validate_existing_cutout(
            partial_path, target, size_kpc, allow_uninitialized=True
        )

    counts = np.zeros(6, dtype=np.uint64)
    with h5py.File(partial_path, "a", rdcc_nbytes=1024 * 1024) as destination:
        destination.attrs["Exporter"] = "tng50-halo-cutout-exporter"
        destination.attrs["SnapshotNumber"] = target.snapshot
        destination.attrs["SubhaloID"] = target.subhalo
        destination.attrs["ParentHaloID"] = target.parent_halo
        destination.attrs["CutoutCenterComovingKpcOverH"] = center
        destination.attrs["CutoutSizePhysicalKpc"] = size_kpc
        destination.attrs["CutoutGeometry"] = "axis-aligned cube"
        destination.attrs["CoordinateUnits"] = "comoving kpc/h"
        destination.attrs["MemoryBudgetMiB"] = memory_budget_mib
        destination.flush()
        copy_snapshot_metadata(source_path, destination)

        for particle_type, fields in PARTICLE_FIELDS.items():
            label = f"PartType{particle_type}"
            total, ranges = halo_file_ranges(il, base_path, target, particle_type)
            if total == 0:
                print(f"  {label}: parent halo has no particles", flush=True)
                continue

            print(
                f"  {label}: scanning {total:,} parent-halo particles in bounded chunks",
                flush=True,
            )
            mask = None
            mask_path = None
            try:
                mask, mask_path, selected_count = build_selection_mask(
                    ranges,
                    total,
                    particle_type,
                    target,
                    center,
                    box_size,
                    half_width,
                    memory_budget_bytes,
                    work_dir,
                    min_free_gib,
                )
                counts[particle_type] = selected_count
                print(
                    f"    selected {selected_count:,}/{total:,} particles inside "
                    f"the {size_kpc:g} physical kpc cube",
                    flush=True,
                )
                if selected_count:
                    for field in fields:
                        print(f"    streaming {field}", flush=True)
                        write_field_from_mask(
                            destination,
                            ranges,
                            mask,
                            selected_count,
                            particle_type,
                            field,
                            target,
                            memory_budget_bytes,
                            work_dir,
                            min_free_gib,
                        )
            finally:
                if mask is not None:
                    del mask
                if mask_path is not None:
                    mask_path.unlink(missing_ok=True)
                gc.collect()
            print(f"    process peak RSS so far: {peak_rss_mib():.1f} MiB", flush=True)

        set_particle_counts(destination, counts)
        destination.flush()

    os.replace(partial_path, final_path)
    print(f"DONE {final_path}", flush=True)
    return final_path


def load_status(path: Path) -> dict:
    if not path.exists():
        return {"completed": {}, "failed": {}}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", default=DEFAULT_SIMULATION)
    parser.add_argument("--base-path", help="Path ending in TNG50-1/output")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", type=Path, default=TARGETS_FILE)
    parser.add_argument(
        "--size-kpc",
        type=float,
        default=DEFAULT_SIZE_KPC,
        help="Physical side length of the galaxy-centered cube (default: 300)",
    )
    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        default=DEFAULT_MEMORY_BUDGET_MIB,
        help="Maximum working-array budget; chunk sizes are derived from it (default: 256)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for disk-backed masks/buffers (default: OUTPUT_DIR/.work)",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=DEFAULT_MIN_FREE_GIB,
        help="Free-space reserve maintained on output and work filesystems (default: 5)",
    )
    parser.add_argument(
        "--only",
        nargs=2,
        type=int,
        metavar=("SNAPSHOT", "SUBHALO"),
        action="append",
        help="Export only this galaxy; may be repeated",
    )
    parser.add_argument(
        "--limit", type=int, help="Process only the first N selected targets"
    )
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dependencies_available = np is not None and h5py is not None
    if args.size_kpc <= 0:
        raise ValueError("--size-kpc must be positive")
    if args.memory_budget_mib < 32:
        raise ValueError("--memory-budget-mib must be at least 32")
    if args.min_free_gib < 0:
        raise ValueError("--min-free-gib cannot be negative")

    targets = load_targets(args.targets)
    if args.only:
        by_key = {(target.snapshot, target.subhalo): target for target in targets}
        requested_keys = [tuple(pair) for pair in args.only]
        unknown = set(requested_keys) - set(by_key)
        if unknown:
            raise ValueError(
                f"Requested snapshot/subhalo pairs are not in {args.targets}: "
                f"{sorted(unknown)}"
            )
        targets = [by_key[key] for key in requested_keys]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        targets = targets[: args.limit]

    if args.list_targets or args.dry_run:
        for target in targets:
            print(f"{target.snapshot},{target.subhalo},{target.parent_halo}")
        parent_count = len(
            {(target.snapshot, target.parent_halo) for target in targets}
        )
        print(
            f"{len(targets)} galaxies in {parent_count} unique parent halos",
            file=sys.stderr,
        )
        return 0

    if not dependencies_available:
        raise RuntimeError(
            "h5py and numpy are required for export. They are preinstalled in "
            "the IllustrisTNG Lab; elsewhere, run: pip install -r requirements.txt"
        )

    base_path = locate_base_path(args.simulation, args.base_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (
        args.work_dir.expanduser().resolve() if args.work_dir else output_dir / ".work"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "generation_status.json"
    status = load_status(status_path)

    print(f"Simulation data: {base_path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Disk-backed work directory: {work_dir}", flush=True)
    print(f"Targets: {len(targets)}", flush=True)
    print(f"Cutout: {args.size_kpc:g} physical kpc cube", flush=True)
    print(f"Working-array memory budget: {args.memory_budget_mib} MiB", flush=True)

    failures = 0
    for index, target in enumerate(targets, start=1):
        print(
            f"[{index}/{len(targets)}] snapshot={target.snapshot} "
            f"subhalo={target.subhalo} parent_halo={target.parent_halo}",
            flush=True,
        )
        try:
            result = export_target(
                base_path,
                output_dir,
                args.simulation,
                target,
                args.size_kpc,
                args.memory_budget_mib,
                work_dir,
                args.min_free_gib,
            )
            status["completed"][target.key] = {
                "path": str(result),
                "size_kpc": args.size_kpc,
                "memory_budget_mib": args.memory_budget_mib,
                "updated_at": utc_now(),
            }
            status["failed"].pop(target.key, None)
        except Exception as exc:
            failures += 1
            status["failed"][target.key] = {
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": utc_now(),
            }
            print(f"FAILED {target.key}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if args.fail_fast:
                save_status(status_path, status)
                raise
        finally:
            save_status(status_path, status)

    print(
        f"Finished: {len(targets) - failures} succeeded/skipped, {failures} failed. "
        f"Status: {status_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
