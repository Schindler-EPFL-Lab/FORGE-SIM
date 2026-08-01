"""
Tests for `SplineToFEMProjector`, the spline -> FEM-mesh bridge.

The projector is the piece that decides *what geometry and what fields the FEM
solver sees*, so an error here is invisible downstream: the solve still runs, the
energy still balances, and the answer is simply the answer to a different
problem. The tests are therefore built around closed forms rather than smoke
checks.

The central one is a manufactured solution: an affine field of the control points
must evaluate at every mesh vertex to the same affine function of that vertex's
position. That single equality exercises the texture-coordinate -> parametric
mapping, the per-patch evaluation and the local->global scatter simultaneously,
across all six patches of the fixture, against a formula that owes nothing to the
implementation.

The rest pin the two-material path (through invariants that need no knowledge of
the transition shape, which is a deliberate approximation), the length-scaling
contract, and the constructor's validation -- including the six-DOF guard that
stops a Reissner-Mindlin shell spline from being read as a temperature field.

Run from the repository root::

    conda run -n dolfinx-rebel-shape python -m unittest discover dolfinx/tests
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _spline_fixtures import (
    affine_field,
    make_iga_spline_file,
    shell_field,
    upper_hemisphere_mask,
)
from spline_to_fem_projector import SplineToFEMProjector

# Field of the control points used by most tests: strictly positive on the unit
# sphere (its range is [1, 3] there), so "every vertex got a value" is testable
# by a positivity check -- an unwritten vertex would keep its zero initialisation.
_AFFINE = affine_field(a=0.5, b=-0.25, c=1.0, d=2.0)

# Two-material properties. The five fields span five orders of magnitude
# (thickness 0.01 vs density 2500) and two of them move in opposite directions
# between the materials, so a mis-wired `dr.select` cannot hide.
_CONCRETE = {
    "conductivity": 2.0,
    "density": 2250.0,
    "specific_heat": 900.0,
    "thickness": 0.2,
    "source": 0.0,
}
_GLASS = {
    "conductivity": 0.9,
    "density": 2500.0,
    "specific_heat": 840.0,
    "thickness": 0.01,
    "source": 10.0,
}


def _project_single_material(projector: SplineToFEMProjector):
    """
    Runs the single-material projection with the concrete properties. The
    material values are irrelevant to every test that uses it -- they are copied
    verbatim into constant arrays -- but the signature requires them, so naming
    them once here keeps the tests about what they are testing.
    """
    return projector.project_fields(
        conductivity=_CONCRETE["conductivity"],
        density=_CONCRETE["density"],
        specific_heat=_CONCRETE["specific_heat"],
        thickness=_CONCRETE["thickness"],
        source=_CONCRETE["source"],
    )


class TestSplineToFEMProjector(unittest.TestCase):
    """
    Projection of spline fields onto the tessellation vertices.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Building a cubed sphere costs ~0.5 s, so the three fixtures are built
        # once and shared; every test below is read-only on them.
        cls._directory = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls._directory, ignore_errors=True)

        cls.thermal_path = make_iga_spline_file(
            cls._directory / "thermal.json", _AFFINE
        )
        cls.mask_path = make_iga_spline_file(
            cls._directory / "mask.json", upper_hemisphere_mask
        )
        cls.shell_path = make_iga_spline_file(
            cls._directory / "shell.json", shell_field
        )

        cls.projector = SplineToFEMProjector(
            thermal_spline_path=cls.thermal_path, tessellation_subdivisions=8
        )
        cls.mesh_data = cls.projector.get_mesh_data()
        cls.two_material_data = cls.projector.project_two_material_fields(
            material_mask_spline_path=cls.mask_path,
            material_threshold=0.5,
            concrete_conductivity=_CONCRETE["conductivity"],
            concrete_density=_CONCRETE["density"],
            concrete_specific_heat=_CONCRETE["specific_heat"],
            concrete_thickness=_CONCRETE["thickness"],
            glass_conductivity=_GLASS["conductivity"],
            glass_density=_GLASS["density"],
            glass_specific_heat=_GLASS["specific_heat"],
            glass_thickness=_GLASS["thickness"],
            glass_source=_GLASS["source"],
        )

    def test_affine_field_of_control_points_is_reproduced_at_every_vertex(self):
        """
        Manufactured solution: with the control-point field set to
        `0.5x - 0.25y + z + 2`, the projected value at every mesh vertex equals
        that same expression evaluated at the vertex position.

        The identity holds because geometry and field share one basis: the
        vertex position is `sum_i N_i(u) P_i` and the field is
        `sum_i N_i(u) L(P_i)` for affine `L`, and `sum_i N_i = 1`, so the second
        is `L` of the first. Any error in the texcoord -> parametric mapping,
        in the per-patch evaluation, or in the local->global scatter breaks it
        at the affected vertices.

        `atol=1e-6` is ~10 ulp of float32 (Mitsuba's `llvm_ad_mono` evaluates the
        basis in single precision) accumulated over the (p+1)^2 = 16 active basis
        terms, on values of order 3. Measured maximum: 6.0e-8.
        """
        temperature = _project_single_material(self.projector).temperature
        vertices = self.mesh_data.vertices

        expected = 0.5 * vertices[:, 0] - 0.25 * vertices[:, 1] + vertices[:, 2] + 2.0
        np.testing.assert_allclose(temperature, expected, atol=1e-6)

    def test_every_vertex_receives_a_projected_value(self):
        """
        No vertex is left at its zero initialisation -- in particular none of the
        vertices on the five patch seams, which are written by more than one
        patch and are exactly where a mismatched local->global mapping would
        leave holes.

        The fixture field is bounded below by 1 on the unit sphere, so a single
        unwritten vertex shows up as a zero. The shape and face-index checks are
        here for the same reason: they are what "the mesh the FEM solver will be
        built on" means.
        """
        field_data = _project_single_material(self.projector)
        n_vertices = len(self.mesh_data.vertices)

        self.assertGreater(field_data.temperature.min(), 0.0)
        for name in ("temperature", "conductivity", "density", "thickness", "source"):
            with self.subTest(field=name):
                self.assertEqual(getattr(field_data, name).shape, (n_vertices,))
        self.assertLess(self.mesh_data.faces.max(), n_vertices)
        self.assertEqual(len(np.unique(self.mesh_data.faces)), n_vertices)

    def test_six_dof_shell_spline_is_rejected(self):
        """
        Regression: a Reissner-Mindlin shell spline (six values per control
        point) is refused, with a message naming the DOF count.

        Reading its first column instead yields a plausible-looking temperature
        field -- typically identically zero, which gives a spatially uniform
        "solution" in which diffusion does nothing and the run still completes,
        reports conserved energy, and writes a file. That is precisely the kind
        of failure a test has to catch, because nothing downstream will.
        """
        projector = SplineToFEMProjector(
            thermal_spline_path=self.shell_path, tessellation_subdivisions=4
        )

        with self.assertRaises(ValueError) as raised:
            _project_single_material(projector)

        self.assertIn("6", str(raised.exception))

    def test_length_scaling_normalises_the_control_point_box(self):
        """
        `target_length_scale` sets the longest **control-point** bounding-box
        axis to the requested value -- not the longest axis of the surface, which
        the B-spline convex-hull property makes strictly smaller (measured: 6.53
        for a target of 7.0 on this sphere).

        This is a "do not fix this" test. `_apply_length_scaling` is a verbatim
        parity port of the IGA runner (`rebel_shape/scripts/heat_simulation/
        heat_simulation.py`), which also normalizes the control-point box;
        "correcting" either side alone to the surface box would silently
        desynchronise the geometry the two solvers integrate and invalidate every
        published IGA/FEM comparison.

        The projected field must be unchanged by the scaling: it is intrinsic to
        the parametrisation, and the control-point values are not touched.
        """
        target = 7.0
        projector = SplineToFEMProjector(
            thermal_spline_path=self.thermal_path,
            tessellation_subdivisions=8,
            target_length_scale=target,
        )

        control_points = projector.thermal_spline.global_control_points.numpy()
        control_extent = control_points.max(axis=0) - control_points.min(axis=0)
        # Exact by construction; the 1e-5 absolute slack is the float32 storage
        # of the scaled control points (7.0 * 1.2e-7 = 8.4e-7 per coordinate).
        self.assertAlmostEqual(float(control_extent.max()), target, delta=1e-5)

        vertices = projector.get_mesh_data().vertices
        surface_extent = vertices.max(axis=0) - vertices.min(axis=0)
        self.assertLess(surface_extent.max(), target)

        np.testing.assert_allclose(
            _project_single_material(projector).temperature,
            _project_single_material(self.projector).temperature,
            atol=1e-6,
        )

    def test_length_scaling_is_a_pure_similarity(self):
        """
        The rescaling multiplies every distance by one common factor: it is a
        uniform scaling, not a per-axis normalisation that would distort the
        surface (and change its spectrum, hence the relaxation time, hence every
        normalized time step derived from it).

        Checked on the full pairwise distance matrix of a vertex subsample, so an
        anisotropic scaling in any single axis fails.
        """
        target = 7.0
        projector = SplineToFEMProjector(
            thermal_spline_path=self.thermal_path,
            tessellation_subdivisions=8,
            target_length_scale=target,
        )

        sample = np.arange(0, len(self.mesh_data.vertices), 7)
        reference = self.mesh_data.vertices[sample]
        scaled = projector.get_mesh_data().vertices[sample]

        reference_distances = np.linalg.norm(
            reference[:, None, :] - reference[None, :, :], axis=-1
        )
        scaled_distances = np.linalg.norm(
            scaled[:, None, :] - scaled[None, :, :], axis=-1
        )
        off_diagonal = ~np.eye(len(sample), dtype=bool)
        ratios = scaled_distances[off_diagonal] / reference_distances[off_diagonal]

        # float32 spline evaluation on both sides; 1e-5 relative spread is two
        # orders above the observed scatter and four below any real anisotropy.
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-5)

    def test_two_material_fields_share_one_interpolated_indicator(self):
        """
        All five material fields are the same interpolated indicator, affinely
        remapped: `(f - f_concrete) / (f_glass - f_concrete)` is one and the same
        array for conductivity, density, specific heat, thickness and source.

        That follows from the construction -- the control-point values are
        `concrete + mask * (glass - concrete)` and interpolation is linear -- and
        it is the cheapest complete check of the two-material path: any one of
        the five `dr.select` calls wired to the wrong mask, the wrong material,
        or the wrong branch breaks it immediately. It needs no knowledge of the
        transition shape, which is the deliberate approximation the module
        docstring forbids changing.

        `atol=1e-5` is the float32 evaluation floor (~1e-7 relative) magnified by
        the subtractive normalisation across fields spanning 0.01 to 2500;
        measured agreement is 2.8e-7.
        """
        indicators = {
            name: (getattr(self.two_material_data, name) - _CONCRETE[name])
            / (_GLASS[name] - _CONCRETE[name])
            for name in _CONCRETE
        }
        reference = indicators["conductivity"]
        for name, indicator in indicators.items():
            with self.subTest(field=name):
                np.testing.assert_allclose(indicator, reference, atol=1e-5)

    def test_two_material_values_stay_between_the_two_materials(self):
        """
        Every interpolated material value lies within the interval spanned by the
        two materials, whichever of them is the larger.

        This is the B-spline convex-hull property applied to the control-point
        values, and it is what makes the smooth transition zone physically
        admissible: an overshoot would put negative conductivity or negative
        thickness into the assembly, where it would show up only as a
        mysteriously unstable solve.
        """
        for name in _CONCRETE:
            with self.subTest(field=name):
                low, high = sorted((_CONCRETE[name], _GLASS[name]))
                values = getattr(self.two_material_data, name)
                # Cubic B-splines are convex-hull bounded exactly, so the only
                # admissible excursion is float32 round-off *of the value*
                # (measured: 2e-7 relative, i.e. ~1 ulp). Scaling the slack by
                # the value rather than by the span keeps it four orders below
                # the material contrast it has to detect.
                slack = 1e-6 * max(abs(low), abs(high))
                self.assertGreaterEqual(values.min(), low - slack)
                self.assertLessEqual(values.max(), high + slack)

    def test_two_material_interiors_recover_the_pure_materials(self):
        """
        Away from the mask's transition zone the interpolation returns the pure
        material: deep inside the upper cap (`z > 0.9`) every field is glass, and
        deep inside the lower cap (`z < -0.9`) every field is concrete.

        Together with the previous test this brackets the transition without
        pinning its shape: the two caps fix the values, the convex-hull bound
        fixes the range in between. A threshold applied to *evaluated* values
        rather than control-point values, or an inverted comparison, breaks this.
        """
        z = self.mesh_data.vertices[:, 2]
        upper, lower = z > 0.9, z < -0.9
        self.assertTrue(upper.any() and lower.any())

        for name in _CONCRETE:
            with self.subTest(field=name):
                values = getattr(self.two_material_data, name)
                scale = max(abs(_GLASS[name]), abs(_CONCRETE[name]))
                np.testing.assert_allclose(
                    values[upper], _GLASS[name], atol=1e-6 * scale
                )
                np.testing.assert_allclose(
                    values[lower], _CONCRETE[name], atol=1e-6 * scale
                )

    def test_constructor_validates_its_arguments(self):
        """
        The constructor rejects a missing spline file, a tessellation resolution
        below one, and a non-positive `target_length_scale` -- each of which
        would otherwise fail much later, inside Mitsuba or the solve.
        """
        with self.assertRaises(FileNotFoundError):
            SplineToFEMProjector(thermal_spline_path=self._directory / "absent.json")

        for subdivisions in (0, -3):
            with self.subTest(tessellation_subdivisions=subdivisions):
                with self.assertRaises(ValueError):
                    SplineToFEMProjector(
                        thermal_spline_path=self.thermal_path,
                        tessellation_subdivisions=subdivisions,
                    )

        for scale in (0.0, -1.0):
            with self.subTest(target_length_scale=scale):
                with self.assertRaises(ValueError):
                    SplineToFEMProjector(
                        thermal_spline_path=self.thermal_path,
                        tessellation_subdivisions=4,
                        target_length_scale=scale,
                    )


if __name__ == "__main__":
    unittest.main()
