# `fem/`: FEM cross-validation reference

DOLFINx (FEniCSx) finite-element solvers that mirror the isogeometric (IGA)
solvers in `rebel_shape/isogeometric_analysis/`. Their purpose is **validation**:
the same physical problem is solved on the same geometry by two independent
discretisations, and the answers are compared. A disagreement points at a bug in
one of them; agreement is evidence for both.

## This is not part of the installable package

`fem/` is deliberately outside `rebel_shape/`:

- `pyproject.toml` declares **no** DOLFINx dependency, and nothing in
  `rebel_shape/` imports from this directory.
- No test in `tests/` covers it, and `python -m unittest discover tests` never
  touches it.
- It needs its **own conda environment**: DOLFINx is a conda-only ecosystem
  whose VTK requirements conflict with the main environment's (`environment.yml`
  substitutes `cadquery-ocp-novtk` for `cadquery-ocp` to work around exactly
  that).

Treat the scripts here as reference implementations that are *run*, not as
library code that is imported by the rest of the project.

## Environment

```bash
conda env create -f fem/environment.yml   # creates `dolfinx-rebel-shape`
conda activate dolfinx-rebel-shape
```

The environment pins `fenics-dolfinx`, `petsc4py`, `slepc4py`, `mpi4py` and
`gmsh`, plus the core rebel-shape pip dependencies needed to load and
tessellate B-spline surfaces. SLEPc is required: the normalised-time-step path
and the modal solvers use it for shift-invert eigensolves. PETSc must carry
**MUMPS**: both the per-step Crank-Nicolson solve and the relaxation eigensolve
factorise with it and have no fallback, and a PETSc without it is rejected at
solver construction.

Run every script **from the repository root**, so that `rebel_shape` resolves and
so that `rebel_shape/__init__.py` can create `outputs/rebel_shape.log`:

```bash
conda run -n dolfinx-rebel-shape python fem/heat/heat_simulation_spline.py --help
```

## Layout

`fem` is a regular package (it has an `__init__.py`), so its modules import each
other absolutely: `from fem.heat.fem_heat_runner import …`. It is *not* on
`sys.path` in either environment, so every entry point prepends the repository
root itself; that preamble spells the depth as `parents[n]`, which is what makes
moving a module between subpackages fail loudly rather than silently resolving to
`fem/`.

| Directory | Contents |
|---|---|
| `fem/` | The shared layer every subpackage may use: the XDMF writer, the spline → FEM projector and its mesh/field data records, the IGA/FEM comparison entry point, the DOLFINx mesh handover and the PETSc factor-solver check. |
| `fem/iga_fem_comparison/` | The pieces behind `compare_iga_fem.py`: `XDMFSolution` (the reader), `VertexMapping` (correspondence between the two vertex orderings), `ErrorNorms` and `ComparisonReport`. |
| `fem/heat/` | Transient heat equation: the Crank-Nicolson solver, the shared runner, and the two spline entry points. |
| `fem/shells/` | Reissner-Mindlin shell: the geometrically exact solver, its modal driver, the spline → shell-mesh bridge, the eigenmode writer and the rigid-body-band report. |
| `fem/meshprep/` | Turning a hand-modelled mesh into an analysis surface: repair and welding, closest-point queries against a reference surface, and the P2 edge layout. Pure NumPy, with Open3D and SciPy imported inside the functions: nothing here imports DOLFINx, `basix` or Open3D at module scope, so its characterization suite runs under both interpreters. |
| `fem/end_to_end/` | The end-to-end validation experiment's FEM side, one directory per topic; see the table below. |
| `fem/tests/` | The whole suite, deliberately **flat**: `unittest` dropped namespace-package discovery in 3.11, so a subdirectory without `__init__.py` would run a subset while reporting green. |

`fem/end_to_end/` follows the same domain-first grouping as
[`rebel_shape/scripts/end_to_end/`](../rebel_shape/scripts/end_to_end/), so the two halves
of a stage read as the pair they are. The four stage directories each hold a thin entry
script, a `*_runner.py` carrying its length, and the report record it returns; the two
comparison directories hold one scene's comparison and the CLI over it.

| Directory | Contents |
|---|---|
| `end_to_end/mesh_quality/` | `analyze_mesh_quality.py` and `MeshQualityRunner`: the triangle-quality census and remesh of the ground-truth mesh, reported as `MeshQualityReport`. |
| `end_to_end/ground_truth_bake/` | `ground_truth_bake.py` and `GroundTruthBakeRunner`: bakes the `.npz` inputs the two ground-truth solves consume, so no Mitsuba or Open3D logic has to reach the conda environment. |
| `end_to_end/heat_gt/` | `fem_heat_gt_mesh.py` and `FEMHeatGTRunner`: the reference transient heat solve, on the ground-truth triangle mesh. |
| `end_to_end/modal_gt/` | `fem_shell_modal_gt_mesh.py` and `FEMShellModalGTRunner`: the same shell solver on a raw ground-truth triangle mesh from an `.npz`. |
| `end_to_end/heat_comparison/` | One scene's IGA-versus-FEM heat comparison, and its `compare_scene.py` CLI. Swept across scenes by `main_scripts/compare_all_scenes.py`. |
| `end_to_end/modal_comparison/` | One scene's IGA-versus-FEM modal comparison, and `SplineModalComparisonReport`, the published record. Swept by `main_scripts/compare_modal_fem_iga.py`. |

The package name `fem` is distinct from DOLFINx's `dolfinx.fem` submodule, so
`from dolfinx import fem` inside these modules keeps binding to the real
submodule.

## Tests

```bash
conda run -n dolfinx-rebel-shape python -m unittest discover fem/tests
```

They live here rather than in the repository's top-level `tests/` because that
suite runs under the uv environment, which has no DOLFINx. Putting them there
would either break it or require a skip-guard, and a skip-guard means the tests
silently never run anywhere, which is worse than having none. `discover tests`
only walks `tests/`, so the two suites never collide.

The suite is built on a coarse gmsh sphere and needs no external data. It runs in
about ten seconds and covers the physics that a careless refactor would break.

Heat: energy conservation under Crank-Nicolson, the analytic uniform-source
heating rate, the constant-field steady state (the `λ₀=0` nullspace noted below),
monotone damping of non-constant modes, **second-order accuracy in `dt`** by
step refinement, `scale_source` linearity, the `dt_over_t_relax` round trip and
the material scalings of `T_relax`, and lossless `snapshot_state`/`load_state`.

Shell: the **steel-sphere anchor**, the same unit sphere the IGA side pins in
the repository's top-level `tests/test_steel_sphere_modal_analysis.py`, so the
two discretisations genuinely cross-validate rather than compare with themselves,
plus the analytic thin-shell clusters at n = 2, 3, 4 and their degeneracies,
the six zero-energy rigid-body modes (`K φ = 0` exactly, at any geometry order),
agreement of the three integration schemes, the drilling guards, the B-bar block
algebra, the spline → curved-mesh conformity, and the on-disk eigenmode format,
which is additionally confirmed against a recorded run when
`REBEL_SHAPE_RECORDED_RUNS` points at one.

`test_crank_nicolson_characterization.py` additionally runs one small
reference scenario (a uniform material, a spatially uniform source, a
non-uniform initial field) and checks its diagnostics against relations
derived from the discretization (the discrete conservation law, the source
integral of a constant coefficient, the monotone decay of the field's
non-constant part) rather than against a committed set of numbers.

### Reproducibility

The linear solve is a direct MUMPS Cholesky, so results are reproducible, but
**only single-threaded**. Measured on a 109,808-vertex run: two invocations agree
bitwise (relative difference exactly 0) under `OMP_NUM_THREADS=1`, and differ by
up to 7.2e-16 without it, because MUMPS' threaded factorization varies its
summation order. So when comparing runs before and after a change, set:

```bash
OMP_NUM_THREADS=1 conda run -n dolfinx-rebel-shape python fem/<script>.py ...
```

and expect exact equality. Without it, allow ~1e-15 relative.

## Heat equation (transient)

The FEM counterpart of `rebel_shape/scripts/heat_simulation/`. The parameters are
deliberately the same on both sides (`dt_over_t_relax` or an explicit time step,
`target_delta_temp` source auto-scaling, `target_length_scale` geometry rescaling,
and the shared `ThermalMaterialConfig`), so a given configuration resolves to a
near-identical physical time step under either discretisation.

| File | Role |
|---|---|
| `heat/fem_heat_problem.py` | `FEMHeatProblem`. The forms, the assembled M and K, the temperature state and the two heat diagnostics. Mirrors `isogeometric_analysis/heat_simulation.py`. |
| `heat/crank_nicolson_fem_scheme.py` | `CrankNicolsonFEMScheme`. Takes the problem, builds `A = M + ½·dt·K` once and reuses it across steps. Mirrors `isogeometric_analysis/crank_nicolson_scheme.py`. |
| `heat/material_functions.py` | `MaterialFunctions`: the five material coefficients the solver integrates, as one named record. |
| `heat/fem_heat_runner.py` | Shared orchestration: mesh construction, solver dispatch, time loop, XDMF export. The analogue of the IGA `HeatSimulationRunner`. |
| `heat/fem_simulation_config.py` | `FEMSimulationConfig`: the time-stepping, output and discretization options `fem_heat_runner` resolves a run from. |
| `heat/snapshot_diagnostics.py` | `SnapshotDiagnostics`: energy bookkeeping for one saved snapshot; also the payload type of `fem_heat_runner`'s `DiagnosticsCallback`. |
| `heat/fem_simulation_diagnostics.py` | `FEMSimulationDiagnostics`: what a completed FEM heat run reports about itself, one `SnapshotDiagnostics` per saved snapshot. Names and writes its own `fem_heat_diagnostics.json` sidecar. |
| `heat/spline_heat_simulation.py` | `SplineHeatSimulationParameters` and `run_spline_heat_simulation`: the CLI knobs, validation and run sequence both entry points share. |
| `heat/heat_simulation_spline.py` | Entry point, single material. |
| `heat/heat_simulation_spline_two_materials.py` | Entry point, two materials (facade mask). |
| `spline_to_fem_projector.py` | `SplineToFEMProjector`: tessellates an IGA spline surface and projects temperature and material fields onto the FEM mesh vertices. |
| `fem_mesh_data.py` | `FEMMeshData`: the triangle-mesh geometry `SplineToFEMProjector` and the ground-truth-mesh drivers hand to DOLFINx mesh creation. |
| `fem_field_data.py` | `FEMFieldData`: the per-vertex temperature and material fields projected onto an FEM mesh. |
| `dolfinx_mesh.py` | `create_manifold_mesh`: hands a serially-built triangle mesh to DOLFINx as a 2-manifold in 3D, gated on rank 0. Shared with the shell side. |
| `petsc_factor_solver.py` | `FactorSolver`: the direct-factorization backends the solvers choose between, valued as PETSc `MatSolverType` strings, and `require_factor_solver`, which checks up front that PETSc was configured with the one a factorization names. |
| `compare_iga_fem.py` | Loads an IGA and a FEM XDMF pair, reports L² and L^∞ error time series, optionally writes a difference series for ParaView. |
| `xdmf_time_series.py` | `write_time_series_xdmf`: the single writer for the custom IGA-compatible XDMF/HDF5 format, shared by the heat runner (solution series) and `compare_iga_fem.py` (difference series). |

Typical loop: run the IGA simulation, run the FEM simulation with the same
parameters, then diff them. Every run's archival output is the XDMF time
series; open it in ParaView. Beside it each FEM run leaves a
`fem_heat_diagnostics.json`: the resolved time step, the relaxation time it came
from, the applied source power, and the per-snapshot energies, none of which the
series itself records. Give both runs the same `save_interval` and
`num_time_steps`: the diff pairs snapshot for snapshot and refuses a mismatch,
since fields at different instants cannot be compared.

```bash
python rebel_shape/scripts/heat_simulation/heat_simulation.py --help          # IGA
conda run -n dolfinx-rebel-shape python fem/heat/heat_simulation_spline.py --help  # FEM
conda run -n dolfinx-rebel-shape python fem/compare_iga_fem.py --help         # diff
```

### A deliberate approximation

Two-material assignment is done *by mesh*: the material mask is thresholded at
control points and spline-interpolated to vertices. This is the intentional
analogue of the IGA solver's evaluation at quadrature points, chosen so the two
discretisations are compared on equal footing. It is not an oversight; please do
not "fix" it.

## Shell modal analysis

The Reissner-Mindlin shell solvers belong to the shell workstream and arrive with
it rather than with the heat pipeline. They validate the IGA modal spectrum in
`rebel_shape/isogeometric_analysis/reissner_mindlin_modal_problem.py`.

| File | Role |
|---|---|
| `shells/geometrically_exact_shell_problem.py` | `GeometricallyExactShellProblem`, the assemble-only half: space, geometry, strains, forms and the B-bar cure. Mirrors the IGA `ReissnerMindlinModalProblem`. |
| `shells/geometrically_exact_shell_modal_solver.py` | `GeometricallyExactShellModalSolver`, the solve-only half: the shift, both eigen paths, and the rigid-body energy diagnostic. Mirrors the IGA `ModalAnalysisSolver`. |
| `shells/shell_system_assembler.py` | `ShellSystemAssembler`, the scipy/PETSc backends behind one `assemble`, with a nested `Backend` enum. |
| `shells/shell_forms.py` | `ShellForms`, the compiled bilinear forms of one configuration, with its `BBar` record nested. |
| `shells/rigid_body_mode.py` | `RigidBodyMode`, the six analytic rigid-body fields the energy diagnostic evaluates. |
| `shells/curved_sphere_mesh.py` | `make_curved_sphere_mesh`, the gmsh benchmark sphere. |
| `shells/shell_element.py` | `ShellElement`, the solver's element and integration choice, whose `LockingCure` and `NormalKind` enums fix the discretisation. |
| `shells/modal_result.py` | `ModalResult`, the eigenpairs record both solve paths return. |
| `shells/rigid_body_mode_energy.py` | `RigidBodyModeEnergy`, the Rayleigh quotient of one of the six analytic rigid-body fields: the solver's own check that its strain measures vanish on them, independent of the eigensolver. |
| `shells/rigid_body_mode_detection.py` | `RigidBodyModeDetection`, the count and gap ratio `detect_rigid_body_modes` returns. |
| `shells/spline_to_shell_mesh.py` | Loads and rescales a spline surface, and builds the order-1 or curved order-2 DOLFINx mesh on its mid-surface. |
| `shells/shell_modal_analysis_spline.py` | Entry point: the CLI parameters and the run sequence, on a spline surface. |
| `shells/eigenmode_xdmf.py` | `export_mode_shapes_xdmf`: the IGA-compatible eigenmode XDMF/HDF5 pair, shared by the spline driver and the ground-truth-mesh one so their output cannot drift apart. |
| `shells/mode_payload.py` | `ModePayload`, one exported mode's arrays, so the HDF5 and XDMF halves of that pair cannot disagree about what a mode carries. |
| `shells/rigid_body_modes.py` | `detect_rigid_body_modes` / `report_rigid_body_modes`: locating the rigid-body band as the largest relative gap in the low spectrum, which is the only check a production run makes on its own output. |
| `end_to_end/modal_gt/fem_shell_modal_gt_mesh.py` | The same solver on a raw ground-truth triangle mesh from an `.npz`. |
