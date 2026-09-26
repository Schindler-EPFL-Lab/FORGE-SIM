# Build context

The Docker build context for the `rebel-shape` Azure ML environment, which is the image
every job in `azure_scripts/` runs in. It starts from a CUDA 12.6 base, installs the build
toolchain that Dr.Jit and SuiteSparse need, and then resolves the Python dependencies with
`uv sync --frozen` and the `azure`, `foreground-background` and `fast-cholesky` extras.

The `pyproject.toml`, `uv.lock` and `.python-version` in this directory are the ones the
image is built from: the `Dockerfile` copies these rather than the repository root's, so
they are maintained here and separately from the root pair.
