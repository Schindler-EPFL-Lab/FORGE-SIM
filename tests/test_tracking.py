import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlflow

from forge_sim import logger as forge_sim_logger
from forge_sim import tracking
from forge_sim.tracking import TrackingConfig, TrackingMode


class TestTracking(unittest.TestCase):
    """
    Tests for `forge_sim.tracking`: `TrackingConfig.from_environment` and its derived
    properties, the module-level singleton (`configure`, `is_enabled`, `run` and the
    `log_*` functions), attaching to a run that a job harness pre-created, and the
    disabled-by-default guarantee.

    The module singleton is reset around every test so that state from one test cannot
    leak into the next, and MLflow is mocked everywhere except where a test says
    otherwise, so that a bug here cannot write to the worktree's `mlruns/`.
    """

    REPO_ROOT = Path(__file__).resolve().parent.parent

    _TRACKING_ENV_VARS = (
        tracking.TRACKING_MODE_VARIABLE,
        tracking.TRACKING_URI_VARIABLE,
        *tracking.AZURE_ML_JOB_VARIABLES,
    )

    @staticmethod
    @contextlib.contextmanager
    def tracking_environment(**overrides):
        """
        Clears the environment variables `TrackingConfig.from_environment` reads and
        applies `overrides` on top, restoring whatever was there before on exit. Used
        instead of `unittest.mock.patch.dict(..., clear=True)` so that unrelated
        variables (`PATH`, `HOME`, ...) are left alone.
        """
        previous = {
            name: os.environ.get(name) for name in TestTracking._TRACKING_ENV_VARS
        }
        for name in TestTracking._TRACKING_ENV_VARS:
            os.environ.pop(name, None)
        os.environ.update(overrides)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def setUp(self):
        tracking._config = None
        tracking._latched_off = False

    def tearDown(self):
        tracking._config = None
        tracking._latched_off = False

    # TrackingConfig.from_environment
    #
    # `TrackingConfig.from_environment` and its derived properties are pure functions
    # of `os.environ`; they never touch the module singleton, so the reset above is
    # simply redundant for the tests in this section.

    def test_missing_or_empty_mode_variable_defaults_to_auto(self):
        """An unset variable and one exported-but-cleared must both mean AUTO, since
        that is how an exported-but-cleared shell variable presents itself."""
        for label, value in (("unset", None), ("empty string", "")):
            with (
                self.subTest(label),
                self.tracking_environment(
                    **(
                        {}
                        if value is None
                        else {tracking.TRACKING_MODE_VARIABLE: value}
                    )
                ),
            ):
                config = TrackingConfig.from_environment()
                self.assertIs(config.mode, TrackingMode.AUTO)

    def test_auto_mode_disabled_when_environment_names_no_destination(self):
        """The core "off by default" guarantee: with every tracking variable unset,
        AUTO must resolve to disabled, since nothing tells it where to report to."""
        with self.tracking_environment():
            config = TrackingConfig.from_environment()
        self.assertIs(config.mode, TrackingMode.AUTO)
        self.assertFalse(config.is_enabled)

    def test_mode_variable_parsed_case_and_whitespace_insensitively(self):
        """'off', 'OFF' and ' Off ' must all resolve to the same mode."""
        cases = (
            ("off", TrackingMode.OFF),
            ("OFF", TrackingMode.OFF),
            (" Off ", TrackingMode.OFF),
            ("on", TrackingMode.ON),
            ("AUTO", TrackingMode.AUTO),
        )
        for raw, expected in cases:
            with (
                self.subTest(raw=raw),
                self.tracking_environment(**{tracking.TRACKING_MODE_VARIABLE: raw}),
            ):
                config = TrackingConfig.from_environment()
                self.assertIs(config.mode, expected)

    def test_malformed_mode_variable_raises_value_error_naming_the_value(self):
        """A `FORGE_SIM_TRACKING` value that is not auto/on/off must raise, since
        `configure` relies on this to decide when to warn and fall back to OFF."""
        with self.tracking_environment(**{tracking.TRACKING_MODE_VARIABLE: "bogus"}):
            with self.assertRaises(ValueError) as context:
                TrackingConfig.from_environment()
        self.assertIn("bogus", str(context.exception))

    def test_empty_tracking_uri_treated_as_unset(self):
        """An exported-but-empty `MLFLOW_TRACKING_URI` must resolve to `None`, not the
        empty string, or `is_enabled` in AUTO mode would see a truthy destination."""
        with self.tracking_environment(**{tracking.TRACKING_URI_VARIABLE: ""}):
            config = TrackingConfig.from_environment()
        self.assertIsNone(config.tracking_uri)

    def test_auto_mode_enabled_by_any_destination_marker(self):
        """AUTO must enable on a tracking URI alone, on `MLFLOW_RUN_ID` alone (the
        variable `mlflow.start_run` itself consults), and on `AZUREML_RUN_ID` alone
        (evidence a job harness is present even though MLflow does not read it)."""
        cases = (
            ("tracking_uri", tracking.TRACKING_URI_VARIABLE, "http://tracking.local"),
            ("mlflow_run_id", "MLFLOW_RUN_ID", "run-123"),
            ("azureml_run_id", "AZUREML_RUN_ID", "azure-run-456"),
        )
        for label, variable, value in cases:
            with self.subTest(label), self.tracking_environment(**{variable: value}):
                config = TrackingConfig.from_environment()
                self.assertTrue(config.is_enabled)

    def test_mode_on_enabled_even_without_any_destination(self):
        """ON must enable tracking even when no destination is named; MLflow's own
        defaults then apply."""
        with self.tracking_environment(**{tracking.TRACKING_MODE_VARIABLE: "on"}):
            config = TrackingConfig.from_environment()
        self.assertTrue(config.is_enabled)

    def test_mode_off_disabled_even_with_tracking_uri_and_job_run_id(self):
        """OFF must win over every destination marker at once - this is the guarantee
        that lets an owner unconditionally silence tracking on a machine that also
        happens to run inside a job harness."""
        with self.tracking_environment(
            **{
                tracking.TRACKING_MODE_VARIABLE: "off",
                tracking.TRACKING_URI_VARIABLE: "http://tracking.local",
                "MLFLOW_RUN_ID": "run-123",
            }
        ):
            config = TrackingConfig.from_environment()
        self.assertFalse(config.is_enabled)

    def test_destination_priority_between_uri_job_marker_and_default(self):
        """`destination` must prefer a named URI over the job-harness phrase, and the
        job-harness phrase over the generic default, so the log line never claims a
        destination that was not actually configured."""
        with (
            self.subTest("uri and job marker both present"),
            self.tracking_environment(
                **{
                    tracking.TRACKING_URI_VARIABLE: "http://tracking.local",
                    "MLFLOW_RUN_ID": "run-123",
                }
            ),
        ):
            config = TrackingConfig.from_environment()
            self.assertEqual(config.destination, "http://tracking.local")

        with (
            self.subTest("only job marker present"),
            self.tracking_environment(**{"MLFLOW_RUN_ID": "run-123"}),
        ):
            config = TrackingConfig.from_environment()
            self.assertEqual(
                config.destination, "the run pre-created by the job harness"
            )

        with (
            self.subTest("neither present but enabled"),
            self.tracking_environment(**{tracking.TRACKING_MODE_VARIABLE: "on"}),
        ):
            config = TrackingConfig.from_environment()
            self.assertEqual(config.destination, "MLflow's own default destination")

    # configure / is_enabled

    def test_malformed_env_warns_and_disables_instead_of_autodetecting(self):
        """A malformed `FORGE_SIM_TRACKING`, discovered through the ordinary
        `is_enabled()` entry point, must warn and disable rather than fall back to
        AUTO - falling back could silently re-enable tracking on a machine whose owner
        deliberately set a (mistyped) value to turn it off."""
        with self.tracking_environment(
            **{
                tracking.TRACKING_MODE_VARIABLE: "bogus",
                tracking.TRACKING_URI_VARIABLE: "http://tracking.local",
            }
        ):
            with self.assertLogs(forge_sim_logger, level="WARNING") as captured:
                enabled = tracking.is_enabled()

        self.assertFalse(enabled)
        self.assertTrue(
            any("bogus" in message for message in captured.output),
            captured.output,
        )

    def test_explicit_config_bypasses_environment_resolution(self):
        """`configure(config)` must install exactly the config it was given, not merge
        it with the environment."""
        with self.tracking_environment(**{tracking.TRACKING_MODE_VARIABLE: "off"}):
            tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        self.assertTrue(tracking.is_enabled())

    def test_latched_tracking_stays_off_after_reconfigure(self):
        """Once a failure has latched tracking off, a fresh `configure` call must not
        re-arm it: the failure is a property of the process, not of the config."""
        tracking._latched_off = True
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        self.assertFalse(tracking.is_enabled())

    def test_is_enabled_auto_configures_from_environment_on_first_call(self):
        """The very first call to `is_enabled()` must resolve the environment on its
        own; a library that only calls `is_enabled()` (never `configure()`) must still
        observe an enabled AUTO configuration."""
        with self.tracking_environment(
            **{tracking.TRACKING_URI_VARIABLE: "http://tracking.local"}
        ):
            self.assertIsNone(tracking._config)
            enabled = tracking.is_enabled()
        self.assertTrue(enabled)

    # log_metric / log_params

    def test_log_metric_failure_latches_and_suppresses_further_calls(self):
        """A failing `mlflow.log_metric` must not raise, and must latch tracking off
        so that a second call does not retry the same broken destination."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with mock.patch.object(
            mlflow, "log_metric", side_effect=RuntimeError("store unreachable")
        ) as log_metric:
            tracking.log_metric("loss", 1.0)
            self.assertFalse(tracking.is_enabled())
            tracking.log_metric("loss", 2.0)
        self.assertEqual(log_metric.call_count, 1)

    def test_log_params_forwards_exact_values_to_mlflow(self):
        """`log_params` must hand its argument to `mlflow.log_params` unchanged - this
        is the seam where a script's resolved configuration crosses into MLflow."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        params = {"learning_rate": 0.01, "optimizer": "lbfgs"}
        with mock.patch.object(mlflow, "log_params") as log_params:
            tracking.log_params(params)
        log_params.assert_called_once_with(params)

    def test_log_metric_forwards_key_value_and_step(self):
        """`log_metric` must forward `step` when given and pass `None` when omitted,
        rather than dropping the keyword entirely."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with mock.patch.object(mlflow, "log_metric") as log_metric:
            tracking.log_metric("loss", 0.5, step=7)
            log_metric.assert_called_once_with("loss", 0.5, step=7)

            log_metric.reset_mock()
            tracking.log_metric("loss", 0.5)
            log_metric.assert_called_once_with("loss", 0.5, step=None)

    def test_log_params_failure_latches_tracking_off(self):
        """A failing `mlflow.log_params` must not raise and must latch, mirroring
        `log_metric`'s contract for the sibling function."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with mock.patch.object(
            mlflow, "log_params", side_effect=RuntimeError("store unreachable")
        ):
            tracking.log_params({"a": 1})
        self.assertFalse(tracking.is_enabled())

    # log_directory

    def test_log_directory_missing_path_never_contacts_mlflow(self):
        """A directory that does not exist must return silently without importing or
        calling MLflow at all."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        missing = Path(tempfile.gettempdir()) / "forge-sim-tracking-test-missing"
        self.assertFalse(missing.exists())
        with mock.patch.object(mlflow, "log_artifacts") as log_artifacts:
            tracking.log_directory(missing)
        self.assertEqual(log_artifacts.call_count, 0)

    def test_log_directory_uploads_existing_directory_with_artifact_path(self):
        """An existing directory must be uploaded as a whole, with `artifact_path`
        forwarded - one call per directory rather than one per file."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "checkpoint.txt").write_text("data")
            with mock.patch.object(mlflow, "log_artifacts") as log_artifacts:
                tracking.log_directory(artifact_dir, artifact_path="checkpoints")
        log_artifacts.assert_called_once_with(
            str(artifact_dir), artifact_path="checkpoints"
        )

    def test_log_directory_permission_error_latches_instead_of_raising(self):
        """A `PermissionError` from `Path.is_dir` must be caught by the same guarded
        block as an MLflow failure and latch tracking off, rather than escaping this
        module - it is deliberately checked inside the `try`, not before it."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        unreadable = mock.Mock(spec=Path)
        unreadable.is_dir.side_effect = PermissionError("denied")
        with mock.patch.object(mlflow, "log_artifacts") as log_artifacts:
            tracking.log_directory(unreadable)
        self.assertEqual(log_artifacts.call_count, 0)
        self.assertFalse(tracking.is_enabled())

    # log_dataset_source

    def test_log_dataset_source_forwards_source_name_and_context(self):
        """`log_dataset_source` must build an HTTP-sourced dataset carrying exactly the
        `url`, `name` and `context` it was given."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with mock.patch.object(mlflow, "log_input") as log_input:
            tracking.log_dataset_source(
                "http://data.local/frames.zip", "vggt_frames", context="eval"
            )

        log_input.assert_called_once()
        args, kwargs = log_input.call_args
        dataset = args[0]
        self.assertEqual(dataset.name, "vggt_frames")
        self.assertEqual(dataset.source.url, "http://data.local/frames.zip")
        self.assertEqual(kwargs["context"], "eval")

    def test_log_dataset_source_records_a_local_path_as_a_filesystem_source(self):
        """Every dataset in this repository is a directory on disk, and every call
        site passes such a path. Wrapping one in `HTTPDatasetSource` records in the
        run's lineage that it was fetched over HTTP, so a non-URL input must resolve
        to a filesystem source carrying the path verbatim."""
        from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource

        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with mock.patch.object(mlflow, "log_input") as log_input:
            tracking.log_dataset_source(
                "dataset/buildinga_spring_vggt", "buildinga_spring"
            )

        log_input.assert_called_once()
        dataset = log_input.call_args[0][0]
        self.assertIsInstance(dataset.source, FileSystemDatasetSource)
        self.assertEqual(dataset.source.uri, "dataset/buildinga_spring_vggt")

    def test_log_dataset_source_failure_latches_tracking_off(self):
        """A failing `mlflow.log_input` must not raise and must latch tracking off."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with mock.patch.object(
            mlflow, "log_input", side_effect=RuntimeError("store unreachable")
        ):
            tracking.log_dataset_source("http://data.local/x.zip", "x")
        self.assertFalse(tracking.is_enabled())

    # run: disabled path

    def test_run_disabled_executes_block_and_propagates_exceptions(self):
        """With tracking off, `run` must be a pure pass-through: the block always
        executes, and an exception it raises must propagate unchanged."""
        tracking.configure(TrackingConfig(mode=TrackingMode.OFF))

        with self.subTest("block executes"):
            executed = False
            with tracking.run():
                executed = True
            self.assertTrue(executed)

        with self.subTest("exception propagates"):
            with self.assertRaises(RuntimeError):
                with tracking.run():
                    raise RuntimeError("boom")

    # run: enabled, mocked MLflow

    def test_run_exception_ends_run_as_failed_and_still_propagates(self):
        """When the wrapped block raises, `run` must end the MLflow run with status
        FAILED and re-raise the original exception unchanged - tracking bookkeeping
        must never mask a real failure."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with (
            mock.patch.object(mlflow, "start_run") as start_run,
            mock.patch.object(mlflow, "end_run") as end_run,
        ):
            with self.assertRaises(RuntimeError) as context:
                with tracking.run():
                    raise RuntimeError("boom")

        self.assertEqual(str(context.exception), "boom")
        start_run.assert_called_once()
        end_run.assert_called_once_with(status="FAILED")

    def test_run_success_ends_run_as_finished_and_skips_artifact_upload(self):
        """A block that completes without error must end the run FINISHED, and without
        an `artifact_directory` no artifact upload may be attempted."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with (
            mock.patch.object(mlflow, "start_run"),
            mock.patch.object(mlflow, "end_run") as end_run,
            mock.patch.object(mlflow, "log_artifacts") as log_artifacts,
        ):
            with tracking.run():
                pass

        end_run.assert_called_once_with(status="FINISHED")
        self.assertEqual(log_artifacts.call_count, 0)

    def test_run_uploads_artifact_directory_even_when_block_raises(self):
        """A crashed run must still keep its partial results: the artifact directory
        must be uploaded before the run is ended, even though the block raised."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "partial.txt").write_text("partial")

            with (
                mock.patch.object(mlflow, "start_run"),
                mock.patch.object(mlflow, "end_run") as end_run,
                mock.patch.object(mlflow, "log_artifacts") as log_artifacts,
            ):
                with self.assertRaises(RuntimeError):
                    with tracking.run(artifact_directory=artifact_dir):
                        raise RuntimeError("boom")

        log_artifacts.assert_called_once_with(str(artifact_dir), artifact_path=None)
        end_run.assert_called_once_with(status="FAILED")

    def test_run_start_failure_latches_and_does_not_contact_broken_store_again(self):
        """When `mlflow.start_run` fails, the block must still execute (tracking is
        never allowed to abort a computation), tracking must latch off, and `end_run`
        must not be attempted against the same broken store."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        executed = False
        with (
            mock.patch.object(
                mlflow, "start_run", side_effect=RuntimeError("unreachable")
            ),
            mock.patch.object(mlflow, "end_run") as end_run,
        ):
            with tracking.run():
                executed = True

        self.assertTrue(executed)
        self.assertFalse(tracking.is_enabled())
        self.assertEqual(end_run.call_count, 0)

    def test_run_end_failure_is_swallowed_without_masking_success(self):
        """When `mlflow.end_run` itself fails, `run` must not raise and must latch
        tracking off, even though the wrapped block succeeded."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with (
            mock.patch.object(mlflow, "start_run"),
            mock.patch.object(
                mlflow, "end_run", side_effect=RuntimeError("unreachable")
            ),
        ):
            with tracking.run():
                pass

        self.assertFalse(tracking.is_enabled())

    def test_run_config_parameter_overrides_environment_for_this_call(self):
        """Passing `config=` to `run` must take effect instead of the environment, even
        though the environment alone would have enabled AUTO tracking."""
        with self.tracking_environment(
            **{tracking.TRACKING_URI_VARIABLE: "http://tracking.local"}
        ):
            with (
                mock.patch.object(mlflow, "start_run") as start_run,
                mock.patch.object(mlflow, "end_run") as end_run,
            ):
                with tracking.run(config=TrackingConfig(mode=TrackingMode.OFF)):
                    pass

        self.assertEqual(start_run.call_count, 0)
        self.assertEqual(end_run.call_count, 0)

    def test_run_selects_configured_experiment_before_starting(self):
        """When `experiment_name` is set, `run` must select that experiment, and must
        do so before starting the run."""
        config = TrackingConfig(mode=TrackingMode.ON, experiment_name="paper-runs")
        tracking.configure(config)

        manager = mock.Mock()
        with (
            mock.patch.object(mlflow, "set_experiment") as set_experiment,
            mock.patch.object(mlflow, "start_run") as start_run,
        ):
            manager.attach_mock(set_experiment, "set_experiment")
            manager.attach_mock(start_run, "start_run")
            with mock.patch.object(mlflow, "end_run"):
                with tracking.run():
                    pass

        set_experiment.assert_called_once_with("paper-runs")
        self.assertEqual(
            [call[0] for call in manager.mock_calls], ["set_experiment", "start_run"]
        )

    def test_run_does_not_select_experiment_when_none_configured(self):
        """Without an explicit `experiment_name`, `run` must not call
        `mlflow.set_experiment` at all, so the Azure ML routing via
        `MLFLOW_EXPERIMENT_NAME` is never overridden."""
        tracking.configure(TrackingConfig(mode=TrackingMode.ON))
        with mock.patch.object(mlflow, "set_experiment") as set_experiment:
            with (
                mock.patch.object(mlflow, "start_run"),
                mock.patch.object(mlflow, "end_run"),
            ):
                with tracking.run():
                    pass

        self.assertEqual(set_experiment.call_count, 0)

    def test_run_does_not_select_experiment_when_attaching_to_an_existing_run(self):
        """A configured `experiment_name` must be ignored when a job harness has
        already created the run this process attaches to. `mlflow.start_run` raises
        when the experiment selected by `set_experiment` differs from the one the
        pre-created run belongs to, and this module contains its own failures, so
        calling it anyway would silently latch tracking off for the whole process
        instead of reporting anything."""
        tracking.configure(
            TrackingConfig(
                mode=TrackingMode.ON,
                experiment_name="paper-runs",
                in_job_with_existing_run=True,
            )
        )
        with (
            mock.patch.object(mlflow, "set_experiment") as set_experiment,
            mock.patch.object(mlflow, "start_run") as start_run,
            mock.patch.object(mlflow, "end_run"),
        ):
            with tracking.run():
                pass

        self.assertEqual(set_experiment.call_count, 0)
        start_run.assert_called_once()
        self.assertTrue(tracking.is_enabled())

    # disabled by default, in a fresh interpreter
    #
    # Runs `forge_sim.tracking` in a fresh interpreter, which is the only honest way
    # to assert that `mlflow` was never imported: by the time any test in this file
    # runs, `mlflow` is already present in `sys.modules` of the parent process (this
    # module imports it for mocking), so that check can only be made from a clean
    # process. The subprocess runs with its own scratch directory as the working
    # directory (with `PYTHONPATH` pointed at the repo so `forge_sim` still resolves
    # to this worktree), rather than the repo root, so that this test cannot be
    # confused by - or endanger - whatever a real optimization run may have already
    # left behind in the repo's own gitignored `mlruns/`.

    def test_disabled_by_default_never_imports_mlflow_or_writes_to_disk(self):
        """With no tracking environment variables set, `is_enabled()` must be False,
        `mlflow` must never be imported, and neither `mlruns/` nor `mlflow.db` may
        appear next to the script - the failure mode this guards against is a machine
        with no tracking configured silently starting to write local runs."""
        script = (
            "import sys\n"
            "from forge_sim import tracking\n"
            "assert tracking.is_enabled() is False\n"
            "assert 'mlflow' not in sys.modules\n"
            "tracking.log_metric('loss', 1.0)\n"
            "tracking.log_params({'a': 1})\n"
            "tracking.log_directory(__import__('pathlib').Path('.'))\n"
            "assert 'mlflow' not in sys.modules\n"
            "print('OK')\n"
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in self._TRACKING_ENV_VARS
        }
        env["PYTHONPATH"] = str(self.REPO_ROOT)

        with tempfile.TemporaryDirectory() as scratch:
            scratch_dir = Path(scratch)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=scratch_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}",
            )
            self.assertIn("OK", result.stdout)
            self.assertFalse(
                (scratch_dir / "mlruns").exists(), "disabled tracking created mlruns/"
            )
            self.assertFalse(
                (scratch_dir / "mlflow.db").exists(),
                "disabled tracking created mlflow.db",
            )
