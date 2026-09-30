"""
Experiment-tracking seam over MLflow.

Library code calls this module instead of `mlflow` directly, so that a machine with no
tracking infrastructure runs the same scripts unchanged. Tracking is resolved once from
the environment and is off unless the environment says otherwise; see `TrackingConfig`.

Nothing in this module raises. A tracking failure is logged once and latches tracking
off for the remainder of the process, because an unreachable tracking URI was measured
to take roughly four minutes to fail: MLflow retries up to seven times with exponential
backoff, under a 120 s per-request timeout. The invariant this protects is that
bookkeeping can never change or abort a computation.

Configuration is read from the environment and never written back to it. In particular
this module never calls `mlflow.set_tracking_uri`, which both overrides and overwrites
`MLFLOW_TRACKING_URI` for the process and every subprocess it spawns, and would
therefore leave the environment unable to configure anything.

A job harness may configure the tracking destination and pre-create a run for the
process. So as not to override any of that, this module never calls `set_tracking_uri`,
calls `set_experiment` only when an experiment was asked for and no run is waiting to
be attached to, and attaches to the pre-created run through `mlflow.start_run`.
"""

import contextlib
import enum
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from forge_sim import logger

TRACKING_MODE_VARIABLE = "FORGE_SIM_TRACKING"
"""Environment variable selecting a `TrackingMode`, by its lower-case value."""

TRACKING_URI_VARIABLE = "MLFLOW_TRACKING_URI"
"""MLflow's own environment variable naming the tracking destination."""

ATTACHED_RUN_VARIABLES = ("MLFLOW_RUN_ID",)
"""
Environment variables whose presence indicates that a job harness has already created
a run for this process to attach to.

Any one of them means a destination exists even when `MLFLOW_TRACKING_URI` is not
exported. `MLFLOW_RUN_ID` is also what makes `mlflow.start_run` attach to that run.
"""


class TrackingMode(enum.Enum):
    """How `forge_sim.tracking` decides whether to talk to MLflow."""

    AUTO = "auto"
    """
    Enabled only if the environment points somewhere: `MLFLOW_TRACKING_URI` is set, or
    the process is running inside a job that pre-created a run for it.
    """
    ON = "on"
    """Always enabled. Failures are still contained, never raised."""
    OFF = "off"
    """Never enabled. `mlflow` is not imported and not contacted."""


@dataclass(frozen=True, kw_only=True)
class TrackingConfig:
    """
    The resolved tracking state for this process.

    `mode` is the behaviour that was requested and `tracking_uri` is the URI MLflow
    would use, kept only so that it can be named in the log; this class never calls
    `set_tracking_uri`, because an in-code call to it overrides and overwrites
    `MLFLOW_TRACKING_URI` and would make the environment unable to configure the
    process. `in_job_with_existing_run` records that a harness has already created a
    run for this process, which is a destination in its own right even when no tracking
    URI is exported. `experiment_name`, when given, is the MLflow experiment that runs
    are created in; when `None` the experiment is left to MLflow, which honours
    `MLFLOW_EXPERIMENT_NAME`. It is also ignored when `in_job_with_existing_run`
    holds, since the pre-created run already belongs to an experiment and cannot be
    moved into another one.

    Instances are immutable; build one with `from_environment` rather than by hand.
    """

    mode: TrackingMode = TrackingMode.AUTO
    tracking_uri: str | None = None
    experiment_name: str | None = None
    in_job_with_existing_run: bool = False

    @classmethod
    def from_environment(
        cls, *, experiment_name: str | None = None
    ) -> "TrackingConfig":
        """
        Reads `FORGE_SIM_TRACKING`, `MLFLOW_TRACKING_URI` and the job markers listed
        in `ATTACHED_RUN_VARIABLES`, and returns the resulting configuration.
        `experiment_name` overrides the experiment for this process; leave it `None` to
        let MLflow decide.

        The mode is matched case-insensitively and after stripping surrounding
        whitespace, so `off`, `OFF` and ` Off ` all name `TrackingMode.OFF`. An empty
        or unset variable means `TrackingMode.AUTO`. Variables set to the empty string
        are treated as unset throughout, since that is how an exported-but-cleared
        shell variable presents itself.

        No network access and no filesystem access are performed: the result depends
        only on the environment, so an unreachable tracking store cannot delay or break
        start-up.

        :returns: the resolved `TrackingConfig`
        :raises ValueError: if `FORGE_SIM_TRACKING` is set to a value that is not one
            of "auto", "on" or "off"
        """
        requested_mode = os.environ.get(TRACKING_MODE_VARIABLE, "").strip().lower()
        if requested_mode == "":
            mode = TrackingMode.AUTO
        else:
            try:
                mode = TrackingMode(requested_mode)
            except ValueError:
                valid_modes = ", ".join(f"'{member.value}'" for member in TrackingMode)
                raise ValueError(
                    f"{TRACKING_MODE_VARIABLE} is set to '{requested_mode}', which is "
                    f"not a tracking mode. Valid modes are {valid_modes}."
                ) from None

        return cls(
            mode=mode,
            tracking_uri=os.environ.get(TRACKING_URI_VARIABLE) or None,
            experiment_name=experiment_name,
            in_job_with_existing_run=any(
                os.environ.get(variable) for variable in ATTACHED_RUN_VARIABLES
            ),
        )

    @property
    def is_enabled(self) -> bool:
        """
        Whether tracking calls will be attempted. For `TrackingMode.AUTO` this is true
        only when the environment names a destination, either through a tracking URI or
        through a run that a job harness has already created.

        :returns: True if MLflow will be contacted
        """
        if self.mode is TrackingMode.OFF:
            return False
        if self.mode is TrackingMode.ON:
            return True
        return self.tracking_uri is not None or self.in_job_with_existing_run

    @property
    def destination(self) -> str:
        """
        A human-readable description of where runs will be recorded, for the log line
        that `configure` writes. MLflow's own defaults are never relied upon, so an
        enabled configuration that names no URI is reported as such rather than
        guessed at.

        :returns: a short description of the tracking destination
        """
        if self.tracking_uri is not None:
            return self.tracking_uri
        if self.in_job_with_existing_run:
            return "the run pre-created by the job harness"
        return "MLflow's own default destination"


_config: TrackingConfig | None = None
"""The process's tracking state, installed by `configure` on first use."""

_latched_off: bool = False
"""
Set once a tracking call has failed, which disables tracking for the remainder of the
process. A stale destination fails slowly, so retrying it every iteration would cost
more wall-clock time than the computation being tracked.
"""


def is_enabled() -> bool:
    """
    Whether tracking is currently active. This is false before anything has configured
    the module and the environment names no destination, when the resolved
    configuration disables tracking, and after a failure has latched tracking off.

    Configures the module from the environment if nothing has done so yet, so that a
    library call is never silently conditional on a script having configured first.

    :returns: True if subsequent tracking calls will reach MLflow
    """
    if _latched_off:
        return False
    _ensure_configured()
    return _config is not None and _config.is_enabled


def configure(config: TrackingConfig | None = None) -> None:
    """
    Installs `config` as this process's tracking state, defaulting to
    `TrackingConfig.from_environment()`. Calling this is optional: the first tracking
    call configures from the environment if nothing has.

    Logs one line stating whether tracking is on and where it points, so that a run is
    never silently untracked. A malformed `FORGE_SIM_TRACKING` disables tracking and
    is reported as a warning rather than raised, because the alternative - falling back
    to automatic detection - could enable tracking on a machine whose owner tried to
    turn it off.

    Tracking that has already latched off after a failure stays off; a new
    configuration does not re-arm it, since the failure is a property of the process
    rather than of the configuration.
    """
    global _config

    if config is None:
        try:
            config = TrackingConfig.from_environment()
        except ValueError as error:
            logger.warning(f"Experiment tracking disabled: {error}")
            config = TrackingConfig(mode=TrackingMode.OFF)

    _config = config

    if not is_enabled():
        logger.info("Experiment tracking is off; nothing will be recorded to MLflow.")
        return
    logger.info(f"Experiment tracking is on, recording to {config.destination}.")


@contextlib.contextmanager
def run(
    *,
    run_name: str | None = None,
    artifact_directory: Path | None = None,
    config: TrackingConfig | None = None,
) -> Iterator[None]:
    """
    Scopes an MLflow run around the block, or does nothing at all when tracking is
    disabled.

    `run_name` names the run; when omitted MLflow generates one. `artifact_directory`,
    if given, is uploaded once on exit, including when the block raises, so that a
    crashed run still keeps its partial results. This replaces per-file artifact
    logging inside the library: only files written under `artifact_directory` are
    recorded.
    `config` overrides the process configuration for this run, and is the only way to
    set an experiment name in code.

    When enabled, the run is started with `mlflow.start_run`, which attaches to an
    existing run when `MLFLOW_RUN_ID` is set - the mechanism by which a run inside a
    job joins the job's own run rather than creating a second one. Note that
    `mlflow.start_run` consumes that variable, so only the first `run` of a process
    attaches; a second one would create a run of its own. Every script here opens
    exactly one. The run is
    then ended explicitly, with a status of "FAILED" when the block raised and
    "FINISHED" otherwise, unless a tracking failure latched tracking off first, in
    which case it is left as it stands rather than contacting a broken store again.

    This context manager never raises on account of tracking; exceptions from the
    wrapped block propagate unchanged.
    """
    if config is not None:
        configure(config)
    else:
        _ensure_configured()

    if not is_enabled():
        yield
        return

    started = _start_run(run_name=run_name)
    if not started:
        yield
        return

    status = "FINISHED"
    try:
        yield
    except BaseException:
        status = "FAILED"
        raise
    finally:
        if artifact_directory is not None:
            log_directory(artifact_directory)
        _end_run(status)


def log_params(params: Mapping[str, Any]) -> None:
    """
    Records `params` as the run's parameters. Intended for a script's resolved
    configuration, logged once; library code should not call it, because a constructor
    argument is configuration the caller owns rather than an observation of a
    computation.
    """
    if not is_enabled():
        return

    try:
        import mlflow

        mlflow.log_params(dict(params))
    except Exception as error:
        _disable("could not record run parameters", error)


def log_metric(key: str, value: float, *, step: int | None = None) -> None:
    """
    Records the scalar `value` under `key`, optionally at iteration `step`. Safe to
    call inside an optimization loop: when tracking is disabled this returns after one
    boolean check and performs no I/O.
    """
    if not is_enabled():
        return

    try:
        import mlflow

        mlflow.log_metric(key, value, step=step)
    except Exception as error:
        _disable(f"could not record metric '{key}'", error)


def log_directory(directory: Path, *, artifact_path: str | None = None) -> None:
    """
    Uploads the whole of `directory` as run artifacts, optionally under `artifact_path`
    within the run. One call over a directory is preferred to one call per file:
    against a remote store each file is a separate round trip.

    Does nothing when `directory` does not exist. The existence check is inside the
    guarded block because `Path.is_dir` only swallows the "not there" family of
    `OSError` and propagates a permission error, which would otherwise escape this
    module from inside `run`'s cleanup and replace whatever the wrapped block raised.
    """
    if not is_enabled():
        return

    try:
        if not directory.is_dir():
            return

        import mlflow

        mlflow.log_artifacts(str(directory), artifact_path=artifact_path)
    except Exception as error:
        _disable(f"could not record the artifacts in '{directory}'", error)


def log_dataset_source(url: str, name: str, *, context: str = "training") -> None:
    """
    Records the dataset at `url` under `name` as an input of the run, tagged with
    `context`. Keeps `mlflow.data`, the part of MLflow's surface most likely to move
    between releases, out of the scripts.

    `url` is either an `http`/`https` URL or a local filesystem path, and the dataset
    source is chosen to match: every dataset in this repository is a directory on disk,
    and recording one through `HTTPDatasetSource` would state in the run's lineage that
    it was fetched over HTTP.
    """
    if not is_enabled():
        return

    try:
        import mlflow
        from mlflow.data.dataset_source_registry import resolve_dataset_source
        from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
        from mlflow.data.http_dataset_source import HTTPDatasetSource
        from mlflow.data.meta_dataset import MetaDataset

        if urlparse(url).scheme in ("http", "https"):
            source = HTTPDatasetSource(url=url)
        else:
            # Resolving through the registry rather than naming a class: MLflow builds
            # its concrete filesystem sources dynamically per artifact repository, so
            # there is no importable local-path source to instantiate. Restricting the
            # candidates to `FileSystemDatasetSource` keeps the answer a filesystem one.
            source = resolve_dataset_source(
                url, candidate_sources=[FileSystemDatasetSource]
            )
        mlflow.log_input(MetaDataset(source=source, name=name), context=context)
    except Exception as error:
        _disable(f"could not record the dataset input '{name}'", error)


def _ensure_configured() -> None:
    """
    Configures the module from the environment if `configure` has not run yet, so that
    the first tracking call of a process resolves its own state.
    """
    if _config is None:
        configure()


def _start_run(*, run_name: str | None) -> bool:
    """
    Starts an MLflow run named `run_name`, first selecting the configured experiment if
    one was given. Latches tracking off and returns False if MLflow cannot be reached.

    The experiment is not selected when a job harness has already created the run this
    process attaches to: MLflow knows that run's experiment, and `mlflow.start_run`
    rejects an attachment whose active experiment is a different one - which, since
    this module contains its own failures, would silently latch tracking off for the
    whole process rather than reporting a misconfiguration.

    :returns: True if a run was started and the block may report to it
    """
    try:
        import mlflow

        if (
            _config is not None
            and _config.experiment_name is not None
            and not _config.in_job_with_existing_run
        ):
            mlflow.set_experiment(_config.experiment_name)
        mlflow.start_run(run_name=run_name)
    except Exception as error:
        _disable("could not start a run", error)
        return False
    return True


def _end_run(status: str) -> None:
    """
    Ends the active MLflow run with `status`, which is one of MLflow's run states.

    Does nothing once tracking has latched off, which is what keeps the worst case at
    one slow call per process: a destination that died mid-run would otherwise be given
    a second chance here, at roughly four minutes. The cost is that such a run is left
    in the store as RUNNING, which is the better trade against a store that has just
    demonstrated it is unreachable.
    """
    if not is_enabled():
        return

    try:
        import mlflow

        mlflow.end_run(status=status)
    except Exception as error:
        _disable("could not end the run", error)


def _disable(action: str, error: BaseException) -> None:
    """
    Reports that `action` failed with `error` and latches tracking off for the rest of
    the process.

    Every failure mode is caught rather than a specific exception type: a poisoned
    local store raises `MissingConfigException` or `PermissionError`, an unreachable
    server raises `MlflowException`, and a bad SQLite path raises out of SQLAlchemy.
    The set of things MLflow can raise is not enumerable, and none of them is worth
    interrupting a computation for.
    """
    global _latched_off

    _latched_off = True
    logger.warning(
        f"Experiment tracking is now off for this process: {action} ({error})."
    )
