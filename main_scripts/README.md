# `main_scripts/`: cross-environment drivers

`rebel_shape` and `fem` are libraries. They provide generic functionality driven by a CLI,
for one scene at a time, with the caller supplying its own paths. They enumerate no
datasets and hardcode no scene. `main_scripts` is the other half: it holds the list of
scenes, sweeps the library entry points across them, and records the results. It carries
no logic of its own past the scenes; the moment something computes or compares a result,
it belongs to the package that owns the physics.

It is also the only place a single command may span both Python environments: the uv
`.venv` (Mitsuba, IGA) and the conda `dolfinx-rebel-shape` environment (FEM). It does so
by subprocess, never by import, which is what lets the two sides stay independent.
`conda_python.py` is the single home for resolving the DOLFINx interpreter: from
`--conda-python` where a driver offers it, otherwise from the `REBEL_SHAPE_DOLFINX_PYTHON`
environment variable.

The dependency runs one way. Nothing in `rebel_shape/` or `fem/` imports `main_scripts`;
the only imports of it are the tests that exercise its own modules.

Run every command from the repository root.

## Drivers

| Module | Role |
|---|---|
| `run_pipeline.py` | The synthetic end-to-end validation, as twelve ordered stages across both environments. |
| `run_vggt_pipeline.py` | Post-processing over the real VGGT-reconstructed scenes: mask fix, inpainting, heat, modal. Everything it spawns stays inside the venv. |
| `compare_all_scenes.py` | The batch heat comparison, IGA against DOLFINx on the same geometry, scene by scene. |
| `compare_modal_fem_iga.py` | The batch modal comparison, on the very spline the IGA modal step used. |
| `evaluation/eval_all_*.py` | Batch Chamfer and novel-view-synthesis evaluation over the PLY, BuildNet3D and real-scene sets, each with a `_large_steps` variant for the mesh baseline. Each spawns the per-reconstruction scripts under `rebel_shape/scripts/evaluation/`, then that directory's `aggregate_metrics.py` once per scene to write the scene's metrics file; the drivers hold the scene lists and never read a metric themselves. See [`rebel_shape/scripts/evaluation/README.md`](../rebel_shape/scripts/evaluation/README.md) for what the scripts underneath measure. |

Both comparison drivers do the same thing: resolve a scene's profile and paths, then spawn
the matching script under `fem/end_to_end/` with the conda interpreter. The solve, the diff
and the write all happen there.

## Supporting packages

| Directory | Contents |
|---|---|
| `pipeline/` | `run_pipeline`'s parts: the `Stage` enum, `StageCommands` (which builds the two command shapes and assigns each stage to one), and `DatasetOverrides` (which applies only the dataset-identity fields a run actually set). |
| `scenes/` | The scene registry: the `VGGTScene` enum and the per-scene profiles giving each one its directory, its thermal fields and its structural material. |
| `heat_comparison/`, `modal_comparison/` | One scene's comparison run: resolve the profile, spawn the FEM-side script, report the outcome. |
| `vggt_pipeline/` | `run_vggt_pipeline`'s stages, its per-scene run and its failure record. |

## Running the end-to-end pipeline

```bash
python main_scripts/run_pipeline.py
python main_scripts/run_pipeline.py --stages GT FEM_MODAL IGA_MODAL CMP_MODAL
python main_scripts/run_pipeline.py --dry-run
```

The twelve stages, in canonical order, are `MESH`, `DATASET`, `RECONSTRUCT`, `INPAINT`,
`GT`, `FEM_HEAT`, `IGA_HEAT`, `CMP_HEAT`, `FEM_MODAL`, `IGA_MODAL`, `CMP_MODAL` and
`SUMMARIZE`. Each runs in its own subprocess, because they set different global Mitsuba
variants and so must not share one.

`--dry-run` builds every stage's command, selected or not, and checks that its target resolves (a
`-m` module's import spec, or a script's path), reporting each one and raising on the first
that does not, without running anything. Reading the twelve resolved targets is the fastest
way to see how the two environments are stitched together.

`--name`, `--mesh-path`, `--prebuilt-spline-path` and `--output-root` move together and are
forwarded to every stage; they are how a run is pointed at a different target.
[`rebel_shape/scripts/end_to_end/README.md`](../rebel_shape/scripts/end_to_end/README.md)
covers the stages themselves and what each one deposits.
