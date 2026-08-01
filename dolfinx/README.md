# `dolfinx/` — FEM cross-validation reference

DOLFINx (FEniCSx) finite-element solvers that mirror the isogeometric (IGA)
solvers in `rebel_shape/isogeometric_analysis/`. Their purpose is **validation**:
the same physical problem is solved on the same geometry by two independent
discretisations, and the answers are compared. A disagreement points at a bug in
one of them; agreement is evidence for both.

## This is not part of the installable package

`dolfinx/` is deliberately outside `rebel_shape/`:

- `pyproject.toml` declares **no** DOLFINx dependency, and nothing in
  `rebel_shape/` imports from this directory.
- No test in `tests/` covers it, and `python -m unittest discover tests` never
  touches it.
- It needs its **own conda environment** — DOLFINx is a conda-only ecosystem
  whose VTK requirements conflict with the main environment's (`environment.yml`
  substitutes `cadquery-ocp-novtk` for `cadquery-ocp` to work around exactly
  that).

Treat the scripts here as reference implementations that are *run*, not as
library code that is imported by the rest of the project.

## Environment

```bash
conda env create -f dolfinx/environment.yml   # creates `dolfinx-rebel-shape`
conda activate dolfinx-rebel-shape
```

The environment pins `fenics-dolfinx`, `petsc4py`, `slepc4py`, `mpi4py`, `gmsh`
and `pyvista`, plus the core rebel-shape pip dependencies needed to load and
tessellate B-spline surfaces. SLEPc is required — the normalised-time-step path
and the modal solvers use it for shift-invert eigensolves. PETSc must carry
**MUMPS**: both the per-step Crank–Nicolson solve and the relaxation eigensolve
factorise with it and have no fallback, and a PETSc without it is rejected at
solver construction.

Run every script **from the repository root**, so that `rebel_shape` resolves and
so that `rebel_shape/__init__.py` can create `outputs/rebel_shape.log`:

```bash
conda run -n dolfinx-rebel-shape python dolfinx/heat_simulation_spline.py --help
```

> Note: `dolfinx/` has no `__init__.py`. Running `import dolfinx` from the
> repository root in the *main* uv environment silently binds to this folder as a
> PEP 420 namespace package instead of failing, which looks like success but is
> useless. Inside the conda environment the real installed package wins, so the
> intended workflow is unaffected.

## Tests

```bash
conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
```

They live here rather than in the repository's top-level `tests/` because that
suite runs under the uv environment, which has no DOLFINx. Putting them there
would either break it or require a skip-guard — and a skip-guard means the tests
silently never run anywhere, which is worse than having none. `discover tests`
only walks `tests/`, so the two suites never collide.

The suite is built on a coarse gmsh sphere and needs no external data. It runs in
about a second and covers the physics that a careless refactor would break:
energy conservation under Crank–Nicolson, the analytic uniform-source heating
rate, the constant-field steady state (the `λ₀=0` nullspace noted below),
monotone damping of non-constant modes, **second-order accuracy in `dt`** by
step refinement, `scale_source` linearity, the `dt_over_t_relax` round trip and
the material scalings of `T_relax`, and lossless `snapshot_state`/`load_state`.

`test_crank_nicolson_characterization.py` additionally pins a full small run
against the committed `crank_nicolson_golden.json`, so a refactor can prove it
did not move the numbers. Regenerate that fixture only deliberately:

```bash
python dolfinx/tests/test_crank_nicolson_characterization.py --regenerate-golden-fixture
```

### Reproducibility

The linear solve is a direct MUMPS Cholesky, so results are reproducible — but
**only single-threaded**. Measured on a 109,808-vertex run: two invocations agree
bitwise (relative difference exactly 0) under `OMP_NUM_THREADS=1`, and differ by
up to 7.2e-16 without it, because MUMPS' threaded factorization varies its
summation order. So when comparing runs before and after a change, set:

```bash
OMP_NUM_THREADS=1 conda run -n dolfinx-rebel-shape python dolfinx/<script>.py ...
```

and expect exact equality. Without it, allow ~1e-15 relative.

## Heat equation (transient)

The FEM counterpart of `rebel_shape/scripts/heat_simulation/`. The parameters are
deliberately the same on both sides — `dt_over_t_relax` or an explicit time step,
`target_delta_temp` source auto-scaling, `target_length_scale` geometry rescaling,
and the shared `ThermalMaterialConfig` — so a given configuration resolves to a
near-identical physical time step under either discretisation.

| File | Role |
|---|---|
| `crank_nicolson_fem_solver.py` | `CrankNicolsonFEMSolver`. Assembles M and K once, reuses `A = M + ½·dt·K` across steps. Mirrors `isogeometric_analysis/crank_nicolson_scheme.py`. |
| `fem_heat_runner.py` | Shared orchestration: mesh construction, solver dispatch, time loop, XDMF export. The analogue of the IGA `HeatSimulationRunner`. |
| `heat_simulation_spline.py` | Entry point, single material. |
| `heat_simulation_spline_two_materials.py` | Entry point, two materials (facade mask). |
| `spline_to_fem_projector.py` | `SplineToFEMProjector` — tessellates an IGA spline surface and projects temperature and material fields onto the FEM mesh vertices. |
| `compare_iga_fem.py` | Loads an IGA and a FEM XDMF pair, reports L² and L^∞ error time series, optionally writes a difference series for ParaView. |
| `xdmf_time_series.py` | `write_time_series_xdmf` — the single writer for the custom IGA-compatible XDMF/HDF5 format, shared by the heat runner (solution series) and `compare_iga_fem.py` (difference series). |
| `visualization.py` | PyVista viewer (time-slider temperature field). |

Typical loop: run the IGA simulation, run the FEM simulation with the same
parameters, then diff them.

```bash
python rebel_shape/scripts/heat_simulation/heat_simulation.py --help          # IGA
conda run -n dolfinx-rebel-shape python dolfinx/heat_simulation_spline.py --help   # FEM
conda run -n dolfinx-rebel-shape python dolfinx/compare_iga_fem.py --help         # diff
```

### A deliberate approximation

Two-material assignment is done *by mesh*: the material mask is thresholded at
control points and spline-interpolated to vertices. This is the intentional
analogue of the IGA solver's evaluation at quadrature points, chosen so the two
discretisations are compared on equal footing. It is not an oversight — please do
not "fix" it.

## Shell modal analysis

The Reissner–Mindlin shell solvers (`geometrically_exact_shell_solver.py` and its
drivers) belong to the shell workstream and arrive with it rather than with the
heat pipeline. They validate the IGA modal spectrum in
`rebel_shape/isogeometric_analysis/reissner_mindlin_modal_problem.py`.
