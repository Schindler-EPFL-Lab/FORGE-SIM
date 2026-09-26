from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from fem.fem_field_data import FEMFieldData
from fem.fem_mesh_data import FEMMeshData
from rebel_shape import logger
from rebel_shape.meshing.tessellated_iga_spline_surface import (
    TessellatedIGASplineSurface,
)


class SplineToFEMProjector:
    """
    Projects B-spline field data from a tessellated IGA surface onto an FEM mesh.

    Loads a `TessellatedIGASplineSurface` from file (optionally rescaling its geometry
    to a target length scale, matching the IGA scripts) and tessellates it into a
    triangle mesh at the requested resolution. `get_mesh_data` returns that geometry;
    `project_fields` and `project_two_material_fields` evaluate the spline field at each
    mesh vertex's parametric coordinates (obtained via texture-coordinate mapping) and
    return it, for the single- and two-material cases respectively.

    The tessellation resolution should adequately capture the spline geometry.
    For cubic splines with 256x256 knot spans, 512x512 quads per patch (i.e.
    512x512x2 triangles) provides a fair comparison with the IGA discretization.

    For two-material runs the material properties are thresholded at the control points
    and then interpolated by the spline basis at the mesh vertices. This reproduces the
    smooth transition zones produced by the IGA simulation, so the material field that
    the FEM mesh sees is the same approximation the IGA solver integrates, the key
    requirement for a one-to-one IGA/FEM comparison.
    """

    def __init__(
        self,
        thermal_spline_path: Path,
        tessellation_subdivisions: int = 512,
        target_length_scale: float | None = None,
    ) -> None:
        """
        Initializes the projector with a tessellated IGA spline surface.

        `thermal_spline_path` should point to a JSON file containing a
        serialized `TessellatedIGASplineSurface` with temperature data in its
        field values. The spline is re-tessellated at `tessellation_subdivisions`
        subdivisions per patch in each parametric direction, resulting in
        2 * subdivisions^2 triangles per patch. A Mitsuba variant must already
        be set by the caller.

        If `target_length_scale` is provided, the loaded geometry is uniformly
        rescaled so that the longest axis of its control-point bounding box
        equals this value, before tessellation. This mirrors the IGA scripts so
        both solvers operate on identically scaled geometry. Note that the
        tessellated surface is contained in the convex hull of the control
        points, so its own bounding box ends up somewhat below
        `target_length_scale`. The gap is set by how finely the control net is
        refined, not by the shape: it shrinks monotonically as the control
        net is refined, since a denser net hugs the surface more closely.

        :raises FileNotFoundError: If the spline file does not exist.
        :raises ValueError: If `tessellation_subdivisions` < 1 or
            `target_length_scale` is non-positive.
        """
        if not thermal_spline_path.exists():
            raise FileNotFoundError(f"Spline file not found: {thermal_spline_path}")
        if tessellation_subdivisions < 1:
            raise ValueError("`tessellation_subdivisions` must be at least 1.")
        if target_length_scale is not None and target_length_scale <= 0.0:
            raise ValueError("`target_length_scale` must be positive if provided.")

        self._thermal_spline_path = thermal_spline_path
        self._subdivisions = tessellation_subdivisions

        # Load the IGA spline surface and (optionally) rescale before tessellating
        # so the FEM mesh geometry matches the IGA run.
        self._thermal_spline = TessellatedIGASplineSurface.from_file(
            composite_spline_surface_path=thermal_spline_path
        )
        if target_length_scale is not None:
            self._apply_length_scaling(target_length_scale)

        self._tessellation = (
            self._thermal_spline.composite_spline.composite_mesh_tesselation(
                subdivisions_per_patch=tessellation_subdivisions
            )
        )

    def _apply_length_scaling(self, target_length_scale: float) -> None:
        """
        Uniformly rescales the loaded geometry so the longest axis of the
        control-point bounding box equals `target_length_scale`.

        This is a verbatim parity port of the IGA scripts' rescaling, down to
        normalizing on the control points rather than on the tessellated
        surface, so that both solvers see identically scaled geometry. The
        surface lies inside the convex hull of the control points, so its own
        bounding box ends up slightly below `target_length_scale`; that is a
        property of this normalization, not a defect.

        Only the geometry (control-point positions) is scaled; field values are
        intrinsic to the parametrization and unaffected.

        :raises ValueError: If the longest control-point extent is non-positive.
        """
        control_points = self._thermal_spline.global_control_points
        bbox_extent = dr.max(control_points, axis=0) - dr.min(control_points, axis=0)
        longest_extent = float(dr.max(bbox_extent).numpy())
        if longest_extent <= 0.0:
            raise ValueError(
                "Cannot apply `target_length_scale`: longest control-point "
                "bbox extent is non-positive."
            )

        factor = target_length_scale / longest_extent
        self._thermal_spline.global_control_points = control_points * factor
        logger.info(
            f"Applied target length scaling to longest control-point bbox axis: "
            f"{target_length_scale} (factor={factor:.6g})."
        )

    @property
    def thermal_spline(self) -> TessellatedIGASplineSurface:
        """The loaded tessellated IGA spline surface with thermal data."""
        return self._thermal_spline

    def get_mesh_data(self) -> FEMMeshData:
        """
        Extracts triangle-mesh geometry from the tessellated surface.

        The composite-mesh connectivity ensures proper vertex sharing at patch
        boundaries.

        :returns: FEMMeshData containing vertices and faces.
        """
        vertices = self._tessellation.global_vertices.numpy().astype(np.float64)
        faces = self._tessellation.global_faces.numpy().astype(np.uint32)
        return FEMMeshData(vertices=vertices, faces=faces)

    def project_fields(
        self,
        *,
        conductivity: float,
        density: float,
        specific_heat: float,
        thickness: float,
        source: float,
    ) -> FEMFieldData:
        """
        Projects spline field values onto FEM mesh vertices with uniform material
        properties (single-material case).

        Temperature is evaluated from the thermal spline at each mesh vertex's
        parametric coordinates; the material properties are set to the given
        uniform values: `conductivity` [W/(m*K)], `density` [kg/m^3],
        `specific_heat` [J/(kg*K)], `thickness` [m], and `source`, the heat
        source term [W/m^2].

        :returns: FEMFieldData with all fields at mesh vertices.
        """
        temperature = self._project_thermal_field()
        n_vertices = self._tessellation.global_vertices.shape[0]

        return FEMFieldData(
            temperature=temperature,
            conductivity=np.full(n_vertices, conductivity, dtype=np.float64),
            density=np.full(n_vertices, density, dtype=np.float64),
            specific_heat=np.full(n_vertices, specific_heat, dtype=np.float64),
            thickness=np.full(n_vertices, thickness, dtype=np.float64),
            source=np.full(n_vertices, source, dtype=np.float64),
        )

    def project_two_material_fields(
        self,
        material_mask_spline_path: Path,
        *,
        concrete_conductivity: float,
        concrete_density: float,
        concrete_specific_heat: float,
        concrete_thickness: float,
        glass_conductivity: float,
        glass_density: float,
        glass_specific_heat: float,
        glass_thickness: float,
        glass_source: float,
        material_threshold: float = 0.5,
    ) -> FEMFieldData:
        """
        Projects fields with a two-material configuration based on a mask spline.

        This mirrors the IGA two-material approach exactly:

        1. Threshold the mask spline's control-point values (not evaluated
           values) to obtain a binary material mask at the control points.
        2. Assign each material property at the control-point level via
           `dr.select` on that mask.
        3. Interpolate those control-point fields at the mesh vertices with the
           spline basis.

        Thresholding at the control points and then interpolating preserves the
        smooth transition zones the IGA simulation produces: the FEM mesh sees
        the same material approximation the IGA solver integrates.

        The mask spline is assumed to share the thermal spline's parametrization
        (same geometric model, different inpainted modality), so its geometry is
        irrelevant here; only its field values matter.

        `material_mask_spline_path` is the path to the material mask spline.
        The base (non-window) region takes `concrete_conductivity`,
        `concrete_density`, `concrete_specific_heat` and `concrete_thickness`;
        the window region takes `glass_conductivity`, `glass_density`,
        `glass_specific_heat` and `glass_thickness`. The two families carry the
        same units: thermal conductivity [W/(m*K)], density [kg/m^3], specific
        heat [J/(kg*K)] and shell thickness [m]. `glass_source` is the heat
        source control-point value for window regions [W/m^2] (concrete regions
        get 0). `material_threshold` is the threshold distinguishing the two
        materials.

        :returns: FEMFieldData with spatially varying material properties.
        """
        temperature = self._project_thermal_field()

        mask_spline = TessellatedIGASplineSurface.from_file(
            composite_spline_surface_path=material_mask_spline_path
        )

        # Threshold at the control-point level (as the IGA does), then let the
        # spline basis smoothly interpolate the per-material control-point values.
        thresholded_mask = mask_spline.global_field_values > material_threshold
        ones = dr.ones(dtype=mi.TensorXf, shape=mask_spline.global_field_values.shape)

        material_control_points = {
            "conductivity": dr.select(
                thresholded_mask,
                glass_conductivity * ones,
                concrete_conductivity * ones,
            ),
            "density": dr.select(
                thresholded_mask, glass_density * ones, concrete_density * ones
            ),
            "specific_heat": dr.select(
                thresholded_mask,
                glass_specific_heat * ones,
                concrete_specific_heat * ones,
            ),
            "thickness": dr.select(
                thresholded_mask, glass_thickness * ones, concrete_thickness * ones
            ),
            "source": dr.select(thresholded_mask, glass_source * ones, 0.0 * ones),
        }

        interpolated = {
            name: self._evaluate_field_at_vertices(mask_spline, control_points)
            for name, control_points in material_control_points.items()
        }

        return FEMFieldData(temperature=temperature, **interpolated)

    def _evaluate_field_at_vertices(
        self,
        spline: TessellatedIGASplineSurface,
        control_point_values: "mi.TensorXf",
    ) -> npt.NDArray[np.float64]:
        """
        Interpolates `control_point_values`, field values at control points of
        shape (n_control_points, 1), onto the mesh vertices using the basis of
        `spline`, the spline surface whose basis is used for interpolation.

        :returns: Interpolated field values at mesh vertices, shape (n_vertices,)
        """
        field_values = np.zeros(
            self._tessellation.global_vertices.shape[0], dtype=np.float64
        )

        for (spline_patch, mapping), (tessellated_patch, local_to_global) in zip(
            spline.composite_spline.patches_with_mapping,
            self._tessellation.patches_with_mapping,
        ):
            parametric_coords = spline_patch.texcoords_to_parametric_coords(
                tessellated_patch.texcoords
            )

            local_cp_values = control_point_values[mi.TensorXu(mapping)]
            patch_field = spline_patch.interpolating_spline.interpolate(
                values=local_cp_values, parametric_coords=parametric_coords
            )

            field_values[local_to_global] = patch_field.numpy()[:, 0].astype(np.float64)

        return field_values

    def _project_thermal_field(self) -> npt.NDArray[np.float64]:
        """
        Evaluates the thermal spline's own temperature field at the mesh
        vertices.

        This is `_evaluate_field_at_vertices` applied to the spline's own
        control-point field, which is exactly what the per-patch
        `IGASplinePatch.evaluate_field` would do (the global field array is
        assembled from (and agrees exactly with) the per-patch ones), plus
        the thermal-spline check below.

        :returns: Temperature at mesh vertices, shape (n_vertices,)
        :raises ValueError: If the spline carries more than one value per control
            point, which means it is not a thermal spline.
        """
        # A heat spline carries exactly one field value per control point. Shell
        # splines carry six (three displacements + three rotations), and silently
        # reading their first column yields a plausible-looking but meaningless
        # temperature, typically identically zero, giving a spatially uniform
        # "solution" in which diffusion does nothing. Fail loudly instead.
        control_point_values = self._thermal_spline.composite_spline.global_field_values
        num_dofs = control_point_values.shape[1]
        if num_dofs != 1:
            raise ValueError(
                f"Expected a thermal spline with 1 field value per control point, "
                f"but this surface carries {num_dofs}. A {num_dofs}-DOF surface is "
                "most likely a Reissner-Mindlin shell spline, which does not hold a "
                "temperature field."
            )

        return self._evaluate_field_at_vertices(
            self._thermal_spline, control_point_values
        )
