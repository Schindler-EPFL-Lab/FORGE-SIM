# `end_to_end/`: synthetic end-to-end digital-twin validation

This directory holds a dataset-agnostic pipeline that demonstrates the central claim of
the paper: physical simulations performed on the *fully recovered* geometry (isogeometric
analysis on the reconstructed B-spline surface, with fields recovered by image-space
inpainting) match, completely independently, simulations performed on the *ground-truth*
geometry (DOLFINx finite-element analysis on the source triangle mesh, with exact fields).

"Completely independently" means the two sides share nothing: a different geometric
representation (spline vs. mesh), a different solver (IGA vs. FEM), a different
discretisation, and a different provenance for the physical fields (inpainted from rendered
images vs. evaluated analytically on the mesh).

The experiment is driven entirely by [`end_to_end_config.py`](end_to_end_config.py)'s
`EndToEndConfig`. Suzanne is the reference proof-of-concept target: a two-material thermal
twin (aluminium/steel, split left/right) and a single-steel structural shell, but any
closed-ish triangle mesh can be plugged in by overriding `mesh_path` and the material
assignment rule.

## 1. Layout

```
end_to_end/
├── data_paths.py          the repository root, for `main_scripts/` alone
├── end_to_end_config.py   root config, composing the four sub-configs
├── dataset/               synthesize the ground-truth target and render it
├── reconstruction/        recover the spline, then inpaint fields onto it
├── heat/                  IGA heat solve, and its comparison against FEM
├── modal/                 IGA modal solve, alignment, comparison
└── metrics/               cross-cutting recomputation and the final summary
```

The grouping is domain-first because [`fem/`](../../../fem/) already is, so `fem/heat` and
`end_to_end/heat` read as the pair they are. Each config lands with the thing it
configures, and each report dataclass beside the function that returns it. Where a stage's
work is long enough to be worth separating from its CLI, it moves into a `<stage>_runner.py`
beside the entry module (`dataset/dataset_runner.py`, `metrics/summarize_runner.py` and
the rest), leaving the entry module as the parameter translation and the call. There are no
`__init__.py` files: these are namespace packages, and `python -m` takes the dotted subpath
(`...end_to_end.heat.compare_heat`) directly.

The two IGA solve stages need no runner at all. `heat.iga_heat_simulation` and
`modal.iga_modal_analysis` translate `EndToEndConfig` into the parameters of the production
drivers under `forge_sim/scripts/heat_simulation/` and
`forge_sim/scripts/reissner_mindlin_modal_analysis/` and call them, so the pipeline runs
the same two-material heat solve and the same shell modal solve an operator runs by hand.

## 2. Which half runs where

Two environments, because DOLFINx is conda-only and its VTK conflicts with the uv stack:

* **`.venv`** (uv/pip): the IGA side, rendering, reconstruction, inpainting, all
  comparisons and all reporting. Needs `mitsuba` and `cholespy`, plus `open3d`,
  `matplotlib` and `h5py` for the comparison plots and metrics. `scikit-sparse` (the
  `fast-cholesky` extra) is optional and only faster: `isogeometric_analysis/sparse_cholesky.py`
  prefers CHOLMOD where it imports and falls back to `cholespy` where it does not.
* **conda `dolfinx-forge-sim`** (DOLFINx, PETSc/MUMPS, SLEPc, gmsh, plus Open3D and
  `gpytoolbox` for the mesh preparation): the FEM side only, mesh preparation,
  ground-truth field baking, and the two ground-truth solves. Those stages live in
  [`fem/end_to_end/`](../../../fem/end_to_end/); see
  [`fem/README.md`](../../../fem/README.md) for that half's layout.

`main_scripts/run_pipeline.py` orchestrates both by subprocess, which is why neither
side imports the other. It finds the conda interpreter from `--conda-python`, or from the
`FORGE_SIM_DOLFINX_PYTHON` environment variable.

Run every command **from the repository root**: the package writes
`outputs/forge_sim.log` relative to the working directory on import.

## 3. Running it

```bash
python main_scripts/run_pipeline.py
python main_scripts/run_pipeline.py --stages GT FEM_MODAL IGA_MODAL CMP_MODAL
python main_scripts/run_pipeline.py --dry-run
```

Each stage runs in its own subprocess, because the stages set different global Mitsuba
variants and so must not share one. `--dry-run` checks that every stage's target resolves
without running anything.

A single stage can also be run on its own: each venv stage is a `tyro` CLI over the
whole of `EndToEndConfig`, which is how parameter sweeps are done:

```bash
python -m forge_sim.scripts.end_to_end.heat.iga_heat_simulation --help
python -m forge_sim.scripts.end_to_end.heat.iga_heat_simulation \
    --thermal.dt-over-t-relax 1e-6 --output-root outputs/suzanne_dt_sweep
```

A stage reads the *whole* config, so switching target is not one flag: `--name` alone
only renames artifacts. `--name`, `--mesh-path`, `--prebuilt-spline-path` and
`--output-root` move together, which is why `run_pipeline` exposes exactly those four
and forwards them to every stage it spawns. Its `--scene` fills the first two and the
pre-built spline from a registry of the four targets.

**Recovered geometry.** By default the recovered geometry is a pre-optimised spline at
`data/splines/spline_suzanne.json` (config `prebuilt_spline_path`), and `run_pipeline`
then skips the `reconstruct` stage. The dataset's RGB views use a solid grey albedo
(`dataset.solid_albedo = 0.5`) and are illustrative only, since reconstruction is not run.
Set `prebuilt_spline_path = None` to reconstruct from the dataset instead, and
`solid_albedo = None` with it, so the surface is textured with the checkerboard the
reconstruction needs.

A recovered spline is valid only for the scene it was recovered from. `run_pipeline
--scene` therefore uses the spline registered for that scene: Suzanne's for `SUZANNE`,
none for the other three, which makes the run reconstruct, and it forwards the choice
(`--prebuilt-spline-path None` where there is none) to every stage so that no stage falls
back to Suzanne's spline. An explicit `--prebuilt-spline-path` outranks the scene.
`run_pipeline` does not forward `solid_albedo`, so a reconstructing run there uses the
grey-albedo views.

## 4. Stages, and where their outputs land

All outputs go under `output_root`, which defaults to `outputs/suzanne_e2e/`.

| # | Stage | Module / driver | Env | Output dir | Deliverable |
|---|---|---|---|---|---|
| 1 | Mesh quality + remesh | `fem/end_to_end/mesh_quality/analyze_mesh_quality.py` | conda | `01_mesh_quality/` | `mesh_quality_report.json`, `quality_histograms.png`, `*_remeshed.ply` |
| 2a | Synthetic dataset (19 train + 4 eval) | `dataset.dataset` | venv | `02_dataset/` | `images/` (RGBA), `thermal/`, `windows_masks/`, `transforms.json` |
| 2b | Spline reconstruction | `reconstruction.reconstruct_surface` | venv | `02_reconstruction/` | `<name>_spline.json`, `final_patch_textures/` (6 EXR), `final_comparison.png` |
| 3, 4 | Inpaint temperature + material | `reconstruction.inpaint_fields` | venv | `03_temperature/`, `04_material/` | `<name>_temperature.json`, `<name>_material.json`, `Final_comparison.png`, field renders |
| n/a | Bake GT FEM inputs | `fem/end_to_end/ground_truth_bake/ground_truth_bake.py` | conda | `05_fem_heat/`, `07_fem_modal/` | `heat_gt.npz`, `modal_gt.npz` |
| 5 | **GT** heat (FEM) | `fem/end_to_end/heat_gt/fem_heat_gt_mesh.py` | conda | `05_fem_heat/` | `fem_heat_gt.xdmf`, `fem_heat_gt_log.json` |
| 6 | **Recovered** heat (IGA) | `heat.iga_heat_simulation` | venv | `06_iga_heat/` | `iga_heat.xdmf`, `iga_heat_log.json` |
| 6c | Heat comparison | `heat.compare_heat` | venv | `06_heat_compare/` | `heat_comparison.{json,md,png}` |
| 7 | **GT** modal (FEM) | `fem/end_to_end/modal_gt/fem_shell_modal_gt_mesh.py` | conda | `07_fem_modal/` | `modal_analysis_results.npz`, `eigenmodes.{xdmf,h5}`, `fem_modal_gt_log.json` |
| 8 | **Recovered** modal (IGA) | `modal.iga_modal_analysis` | venv | `08_iga_modal/` | `modal_analysis_results.npz`, `xdmf/eigenmodes.{xdmf,h5}` |
| 8c | Modal comparison | `modal.compare_modal` | venv | `08_modal_compare/` | `modal_comparison.{json,md,png}` |
| n/a | Collated summary | `metrics.summarize` | venv | `output_root` | `SUMMARY.md` (stages not yet run are marked so) |

The two FEM solve stages consume the `.npz` baked by `ground_truth_bake.py`: the conda
environment has no Mitsuba or Open3D, so all UV, checkerboard and material logic stays in
the `.venv`.

## 5. Key parameters

These live in `end_to_end_config.py` and its four sub-configs, where each carries its own
docstring; the values below are the defaults the recorded runs used.

* **Dataset**: 19 training views + 4 eval views on a Fibonacci sphere, 512², RGB rendered
  with a solid grey albedo (`solid_albedo = 0.5`) against the checkerboard env-map
  background.
* **Reconstruction** (only when `prebuilt_spline_path = None`): matches the standard
  spline scripts: degree 3, 2 → 32 knot spans by h-refinement at iterations
  `[25, 50, 80, 140]`, 250 L-BFGS shape + Adam texture iterations, 64/16 spp
  (continuous/visibility) against the dataset's 4096-spp reference renders, 512² texture,
  G1 and curvature regularisation, weighted 1.0 and 2e-3 within a composite that is
  itself scaled by 1e-2, so the effective weights are 1e-2 and 2e-5. These three numbers
  are the one thing here that is not a config field; they are set where the losses are
  built, in `reconstruction/reconstruct_surface.py`. If the downstream simulations look
  under-resolved, raise `modal.iga_modal_spans` or `thermal.inpaint_h_refinement`.
* **Geometry scale**: no rescaling is applied anywhere. The GT mesh and the spline are
  already at the same native scale (to ~0.5%), and the pipeline has the GT geometry and
  poses, so the physical size is the geometry's own (Suzanne's longest axis ≈ 3.34 m,
  Spot's ≈ 2.69 m).
* **GT mesh repair**: UV seams welded; Suzanne's two detached eye caps are stitched to
  their socket rims, preserving the eye bulge, rather than discarded and flat-capped; the
  surface is closed and remeshed (Botsch-Kobbelt) to a watertight FEM mesh. The
  before/after equiangle skewness and aspect ratio of that remesh are recorded per run
  in `01_mesh_quality/mesh_quality_report.json`, rather than asserted here.
* **Materials**: two symmetric left/right halves (`material_axis = 0`, split at the
  sagittal plane). The active, source-carrying half (left, `x < centre`, config slot
  `thermal.steel`) is ALUMINIUM: k = 237, ρ = 2700, c = 900, h = 0.01. The passive half
  (right, config slot `thermal.wood`) is STEEL: k = 50, ρ = 7850, c = 500, h = 0.01,
  roughly an 8× diffusivity contrast with aluminium the faster side. The slot names are
  the pipeline's wire protocol, not claims about the substances; see
  `heat/thermal_sim_config.py`. Modal uses uniform steel: E = 210 GPa, ν = 0.3, ρ = 7850,
  **h = 2 mm**, κ = 5/6, so h/L ≈ 6e-4, thin.
* **Heat**: checkerboard initial temperature in [0, 1]; the source is auto-scaled to a
  0.15 K rise on the active half (`thermal.target_delta_temp`); 1000 steps at
  `dt/T_relax = 1e-5`, fine enough to avoid Crank-Nicolson ringing on the sharp
  checkerboard; snapshots every 100. The IGA run reuses the FEM-derived `dt`, so both
  march on one physical clock. Set `thermal.target_delta_temp = None` and
  `heat_source = 0` for pure diffusion.
* **Modal**: 20 modes. The FEM side is B-bar locking-cured (`--use-bbar`, block-wise
  assembly) with geometrically exact **P3** fields on the faceted GT mesh: a convergence
  study found B-bar P2 about 3% too stiff at this thinness, and P3 converged. The IGA side
  uses cubic B-splines refined to a pinned 128 spans per direction
  (`modal.iga_modal_spans`) rather than B-bar, because the locking the IGA side hits at
  this thinness is membrane rather than shear, so refinement is the cure and B-bar would
  not address it; that field's docstring carries the 64-versus-128 study.

## 6. What the reported numbers mean

Every number states what it isolates. Four comparison types recur:

| Tag | Comparison | Isolates | Where |
|---|---|---|---|
| **V** verify | IGA against DOLFINx on the **same** geometry | the numerical method alone | `main_scripts/compare_all_scenes.py` in batch, `fem/compare_iga_fem.py` for one pair |
| **A** anchor | a solver against a known answer | absolute truth | `tests/test_steel_sphere_modal_analysis.py` |
| **G** geometry | the **same** solver on GT vs. recovered geometry | reconstruction error alone | n/a |
| **E** end-to-end | IGA-on-recovered against FEM-on-GT | the whole pipeline (V and G together) | this directory |

This pipeline is **E**: on its own it cannot attribute a discrepancy to geometry rather
than to the solver, which is why the V and A numbers are reported beside it. The A anchor
is the steel sphere, where the l = 2 spheroidal mode's five-fold degeneracy is known from
the physics and the frequency itself is pinned as a regression reference.

### Heat

* **Total energy, generation rate, relaxation time** (`heat.compare_heat`): the
  geometry-robust scalars both solvers log. A direct field diff across two different
  geometries is meaningless, so the comparison is taken over integrated quantities.
  Note what total heat can and cannot show: with Crank-Nicolson and no Neumann flux it is
  conserved by construction, so it reflects recovered area, mean field and source area,
  and is blind to *where* the heat flows. The plot carries a best-effort peak surface
  |∇T| curve beside those scalars, and the two XDMF series remain the way to put the
  fields themselves side by side in ParaView.
* **H1: first heat-operator eigenvalue** (`metrics.recompute_metrics`):
  `λ₁ = 1 / T_relax`, a geometry-intrinsic Laplace-Beltrami scalar both runs already
  logged, so comparing it needs no correspondence and no re-assembly.
* **H2: cross-geometry field L²** (`heat.heat_field_metrics`): the per-timestep,
  area-weighted L² discrepancy between the two full temperature fields, transferred by
  closest-point projection. Each step carries a **floor**, the field's own symmetric
  round-trip L² (FEM → IGA → FEM, with no recovery involved), and what is reported is the
  floor-corrected excess `sqrt(cross² − floor²)`. The floor matters: a near-binary
  checkerboard loses ~12% to the transfer alone at t₀, because a closest-point map across
  two slightly offset meshes flips edge vertices between 0 and 1. So the t₀ number is
  transfer-limited and does not measure inpainting error (the matching hot-area fraction
  does that), while the diffused-field excess is the honest "do the two solutions agree"
  number. Even floor-corrected it carries a `|∇u| · d_proj` term, `d_proj` being the
  projection distance, i.e. the geometry error; that distance is reported next to it. This
  is end-to-end pipeline fidelity, not solver accuracy.

### Modal

* **Frequencies** (`modal.compare_modal`): the first six rigid-body modes are dropped and
  the elastic modes are compared. The headline statistic is the **mean and median**, not
  the max: near-degenerate modes reorder between the two solvers, so a per-mode maximum is
  a pairing artifact rather than a modelling error.
* **M1: MAC** (`metrics.recompute_metrics`): the area-weighted (lumped-mass) modal
  assurance criterion `MAC(i,j) = |φᵢᵀ M ψⱼ|² / [(φᵢᵀ M φᵢ)(ψⱼᵀ M ψⱼ)]` between the two
  sets of displacement mode shapes. The diagonal is the index-paired agreement; the best
  MAC over all partners answers "is this shape recovered *somewhere*?". A high best MAC
  beside a low diagonal is a reordering, not lost shape, which is what licenses reporting
  the mean rather than the max above.
* **M4: degenerate clusters**: within a near-degenerate frequency cluster the individual
  eigenvectors are arbitrary and only the spanned subspace is comparable, so each cluster
  also reports its summed MAC divided by the cluster size. That is the mean cos² of the
  principal angles between the two subspaces exactly when the cluster's modes are
  orthonormal under the same area weight, which the lumped-area weighting makes true only
  approximately; so read it as a subspace-overlap indicator, not as an identity.
* **M2: parity fit**: `f_IGA = a · f_FEM + b` with slope and R². A systematic stiffness,
  mass or scale bias shows up as a slope away from 1 while R² stays near 1; scattered
  shape error shows up the other way round.

### Geometry, and the correlation that closes the loop

`metrics.recompute_metrics` also computes, per mesh, the Chamfer and Hausdorff distances
and the mean and RMS recovered-to-GT surface distance, each as a percentage of the GT
bounding-box diagonal (which is what makes meshes of different physical size comparable),
plus the surface-area ratio. It then reports the Spearman rank correlation of those
geometry measures against the heat and modal errors: the direct evidence for "better
geometry gives better simulation", rather than an assertion of it.

Two things every metric here obeys: all L² and MAC inner products are area-weighted, never
unweighted vertex means; and the surface samplers are seeded, so a re-run of the geometry
metrics reproduces rather than re-draws.

## 7. Recomputing the metrics from finished runs

The solve stages write raw artifacts and the reporting stages read them, so every table
and plot regenerates without re-solving.

Two of the reporting scripts read a **recorded deliverable tree** (one directory per
mesh, holding the finished runs the paper's numbers came from) rather than a live
`output_root`. They take it from `--root`, or from the `FORGE_SIM_RECORDED_RUNS`
environment variable:

```bash
python -m forge_sim.scripts.end_to_end.metrics.recompute_metrics --root <tree>
python -m forge_sim.scripts.end_to_end.heat.heat_field_metrics --root <tree>
```

`recompute_metrics` writes `<tree>/metrics/recompute_metrics.{json,md}` and
`geometry_sim_correlation.json`; `heat_field_metrics` writes
`<tree>/metrics/heat_field_l2.{json,md}`. That tree names its per-mesh directories for
what they hold (`fem_modal_simulation/`, `iga_modal_simulation/`, `heat_comparison/`, …)
rather than by the stage numbers of §4, because it is curated from finished runs rather
than written by the pipeline.

`align_modal_xdmf` is the odd one out: it works on a **live** run root and so takes
`--name` and `--output-root`, defaulting to `outputs/<name>_e2e` for each recorded mesh
in turn.

```bash
python -m forge_sim.scripts.end_to_end.modal.align_modal_xdmf --name suzanne
```

It writes an `eigenmodes_aligned.xdmf`/`.h5` pair beside the IGA modal export, rotating
the IGA modes into the FEM frame for a side-by-side view; the MAC above is computed
from those aligned modes.

For several scenes at once, `main_scripts/compare_all_scenes.py` runs the cross-scene
comparison, and `forge_sim/metrics/shapes.py` holds the shared geometry metrics.

## 8. Notes and caveats

* **GPU PTX bug.** The `cuda_ad_mono` variant (inpainting, IGA modal) can crash with a
  Dr.Jit PTX-compilation error on some driver/toolkit combinations. Pass `--no-use-gpu` to
  `run_pipeline.py`, or to an individual stage, to run the Mitsuba stages on CPU/LLVM.
* The GT analysis meshes keep only the largest connected component: Suzanne's eyeballs are
  separate surfaces and would otherwise contribute 18 rigid-body modes instead of 6, while
  the spline is a single closed surface.
* Absolute elastic frequencies are discretisation-limited at the default resolutions;
  the recorded runs put that at roughly 10%, observed rather than derived from an error
  bound. Both spectra are reported in full, and refinement tightens them.
