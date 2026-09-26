import tempfile
import unittest
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.scripts.configs.structural_material_config import (
    StructuralMaterialConfig,
)
from rebel_shape.scripts.end_to_end.modal.modal_frequencies import modal_frequencies
from rebel_shape.scripts.reissner_mindlin_modal_analysis.reissner_mindlin_modal_runner import (  # noqa: E501
    ReissnerMindlinModalRunner,
)
from rebel_shape.splines.bspline_patch import BSplinePatch


class TestReissnerMindlinModalRunner(unittest.TestCase):
    NUM_MODES = 7
    """Eigenpairs solved for: fewer than the panel's DOFs and more than one."""

    NUM_DOFS = 54
    """
    Shell DOFs of the fixture panel: a degree-2 patch with one span per direction
    carries 3 x 3 control points, each with 6 DOFs. It matches neither `NUM_MODES`
    nor any other array extent here, so an archived eigenvector block of the wrong
    orientation cannot pass the shape check.
    """

    def setUp(self) -> None:
        mi.set_variant("llvm_ad_rgb")

        patch = BSplinePatch.unit_square(
            patch_name="panel", degree=2, u_subdivisions=1, v_subdivisions=1
        )
        factory = CompositeSplineSurfaceFactory("modal_runner_panel")
        factory.add_patch(patch)
        tessellated_spline = TessellatedSplineSurface(
            composite_spline_surface=factory.build(),
            tessellation_subdivisions_per_patch=2,
        )
        iga_spline = TessellatedIGASplineSurface.from_tessellated_spline_surface(
            tessellated_spline_surface=tessellated_spline,
            per_patch_field_values=[
                dr.zeros(dtype=mi.TensorXf, shape=(spline.control_points.shape[0], 6))
                for spline in tessellated_spline.composite_spline
            ],
        )
        # Every material value differs from `StructuralMaterialConfig`'s own default,
        # so the two the archive test reads back cannot agree with a runner that fell
        # through to those defaults. Both are chosen exact in float32 as well, since
        # the runner broadcasts them into an `mi.TensorXf`.
        self.runner = ReissnerMindlinModalRunner(
            tessellated_iga_spline=iga_spline,
            material_config=StructuralMaterialConfig(
                youngs_modulus=7.0e10,
                poisson_ratio=0.31,
                density=2710.0,
                thickness=0.037,
                shear_correction_factor=0.85,
            ),
        )

    def test_archive_holds_the_eigenpair_the_solve_returned(self) -> None:
        """The saved `.npz` carries the eigenpair the run returned, unaltered."""
        self.assertEqual(float(self.runner.density.numpy()[0, 0]), 2710.0)
        self.assertEqual(self.runner.shear_correction_factor, 0.85)

        with tempfile.TemporaryDirectory() as directory:
            output_folder = Path(directory)

            eigenvalues, eigenvectors = self.runner.run_modal_analysis(
                num_modes=self.NUM_MODES, output_folder=output_folder
            )
            archive = np.load(output_folder / "modal_analysis_results.npz")

            self.assertEqual(set(archive.keys()), {"eigenvalues", "eigenvectors"})
            self.assertEqual(eigenvalues.shape, (self.NUM_MODES,))
            self.assertEqual(eigenvectors.shape, (self.NUM_DOFS, self.NUM_MODES))
            self.assertTrue(
                bool(np.all(np.diff(eigenvalues) >= 0.0)),
                msg="The runner documents an ascending spectrum.",
            )
            self.assertGreater(
                float(np.abs(eigenvectors).max()),
                0.0,
                msg="An all-zero eigenvector block would make the comparison vacuous.",
            )
            np.testing.assert_array_equal(archive["eigenvalues"], eigenvalues)
            np.testing.assert_array_equal(archive["eigenvectors"], eigenvectors)
            # The archive the pipeline's own reader is pointed at, read by it.
            np.testing.assert_array_almost_equal(
                modal_frequencies(output_folder / "modal_analysis_results.npz"),
                np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi),
                decimal=12,
            )

    def test_invalid_num_modes(self) -> None:
        """A non-positive mode count is refused before any output is written."""
        with tempfile.TemporaryDirectory() as directory:
            # A directory the runner would have to create, so its own guard order is
            # observable: against one that already exists, its `mkdir` is a no-op
            # whether the guard runs before it or after.
            output_folder = Path(directory) / "modal"

            with self.assertRaises(ValueError) as raised:
                self.runner.run_modal_analysis(
                    num_modes=0, output_folder=output_folder
                )

            self.assertIn(
                "`num_modes` must be a positive integer", str(raised.exception)
            )
            self.assertFalse(output_folder.exists())
