# TNG50 galaxy-region cutout exporter

This repository exports galaxy-centered regions for the 80 missing TNG50-1
targets (in 61 unique parent FoF halos). It is intended to run inside the
[IllustrisTNG JupyterLab](https://www.tng-project.org/data/lab/), where the full
snapshot data and `illustris_python` are already available.

By default, each output is an axis-aligned cube **300 physical kpc across**,
centered on that target galaxy's `SubhaloPos`. Particle coordinates remain in
their original TNG units (`comoving kpc/h`), and periodic box boundaries are
handled when selecting the region. Particles are drawn from the target's parent
FoF halo, so the local circumgalactic environment is retained without writing
the entire halo.

The exporter writes the same particle fields requested from the web cutout API:

- Gas (`PartType0`): `Coordinates`, `Masses`, `GFM_Metallicity`, `Density`,
  `ElectronAbundance`, `InternalEnergy`, and `StarFormationRate`.
- Stars/wind (`PartType4`): `Coordinates`, `GFM_StellarFormationTime`,
  `GFM_InitialMass`, and `GFM_Metallicity`.

## Run in the TNG Lab

Open a terminal in the Lab and clone this repository:

```bash
git clone https://github.com/astroferreira/tng50-halo-cutout-exporter.git
cd tng50-halo-cutout-exporter
python export_missing_halos.py
```

Or run it from a notebook cell:

```python
%cd ~/tng50-halo-cutout-exporter
%run export_missing_halos.py
```

Generated files go into `missing_galaxy_cutouts_300kpc/`. These spatially cropped
files should be much smaller than the roughly 440 GiB full-halo batch, although
the exact total depends on the particle density around each galaxy. Check your
writable-storage quota before starting.

## Useful commands

Preview all targets without touching the simulation data:

```bash
python export_missing_halos.py --list-targets
```

Test one target first:

```bash
python export_missing_halos.py --only 91 101491
```

The two values passed to `--only` are `SNAPSHOT SUBHALO`, not the parent-halo ID.

Choose another physical cube width if needed:

```bash
python export_missing_halos.py --size-kpc 300
```

Use an explicit simulation mount or destination:

```bash
python export_missing_halos.py \
  --base-path ~/sims.TNG/TNG50-1/output \
  --output-dir /path/with/enough/space/missing_galaxy_cutouts_300kpc
```

Run it in a terminal multiplexer if available:

```bash
tmux new -s tng-export
python export_missing_halos.py 2>&1 | tee export.log
```

Detach with `Ctrl-b`, then `d`, and later reconnect with
`tmux attach -t tng-export`.

## Resuming and downloading

The job is safe to restart:

- Finished `.hdf5` files are skipped.
- In-progress files use the `.hdf5.part` suffix.
- Each dataset is first written under a temporary name, so an interrupted field
  is rewritten on the next run.
- `generation_status.json` records completed and failed targets.

To make a single downloadable archive after the run:

```bash
tar -C missing_galaxy_cutouts_300kpc -czf tng50-missing-galaxy-cutouts.tar.gz \
  --exclude='*.part' --exclude='generation_status.json*' .
```

For large outputs, downloading individual completed `.hdf5` files or using
`rsync`/`scp` is generally more reliable than creating one huge browser download.

## Notes

- The source target list is [`targets.csv`](targets.csv).
- Output names have the form
  `cutout_TNG50-1_<snapshot>_<subhalo>_<parent-halo>.hdf5`, matching the existing
  compatibility filenames used by the SKIRT campaign.
- Snapshot `Header`, `Parameters`, and `Configuration` attributes are copied into
  each output, and particle counts are updated for the cutout.
- Each output records its subhalo, parent halo, center, physical size, geometry,
  and coordinate units as HDF5 root attributes.
- The exporter loads one parent-halo field at a time, applies the spatial mask,
  and writes only selected particles. This limits peak memory use even for the
  largest parent halos.
- No TNG API key is needed because this reads the Lab's mounted snapshot data.

See the official TNG documentation for
[`illustris_python` data access](https://www.tng-project.org/data/docs/scripts/)
and the TNG forum guidance on using
[`snapshot.loadHalo()` for Lab cutouts](https://www.tng-project.org/data/forum/topic/887/cutouts/).
