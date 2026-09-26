from collections.abc import Sequence
from pathlib import Path

import drjit as dr
import jsonpickle
import mitsuba as mi
import numpy as np

from rebel_shape.meshing.composite_mesh import CompositeMesh
from rebel_shape.meshing.patch import Patch
from rebel_shape.meshing.patch_manager import PatchManager
from rebel_shape.meshing.regular_mesh import RegularMesh
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF


class RegularCompositeMesh(CompositeMesh):
    """
    A composite mesh whose patches are `RegularMesh` instances and which therefore
    supports `h_refinement`.

    Refinement subdivides each patch by a factor along both grid directions and then
    rebuilds the patch-to-global mappings, averaging weights, and global mesh from
    the refined patches. This mirrors `CompositeSplineSurface.h_refinement` on the
    spline side, but operates on mesh vertices instead of spline control points.

    The base `CompositeMesh` API is otherwise unchanged: `global_vertices`,
    `global_faces`, `global_vertex_normals`, and `update_global_representation`
    behave as on the parent class.

    The `cubed_sphere` classmethod is a convenience builder that produces a 6-patch
    cubed sphere with the same patch layout, side linking, and texture parametrization
    as `splines.cube_initialization.cubed_sphere`, but using only mesh-side
    infrastructure (no spline).
    """

    _patch_list: Sequence[RegularMesh]
    """Array of `RegularMesh` instances that constitute the composite mesh."""

    _CUBE_FACE_TRANSFORMS: list[tuple[str, list[int], list[float], list[float]]] = [
        # (suffix, axis_permutation, scale, translation): copied verbatim from
        # rebel_shape/splines/cube_initialization.py:unit_cube so the resulting
        # mesh patch layout (and therefore link_sides topology) matches the
        # spline cubed sphere exactly.
        ("_left", [2, 0, 1], [1.0, -1.0, 1.0], [-0.5, 0.0, 0.0]),
        ("_right", [2, 0, 1], [1.0, 1.0, 1.0], [0.5, 0.0, 0.0]),
        ("_back", [0, 2, 1], [1.0, 1.0, 1.0], [0.0, -0.5, 0.0]),
        ("_front", [0, 2, 1], [-1.0, 1.0, 1.0], [0.0, 0.5, 0.0]),
        ("_bottom", [0, 1, 2], [-1.0, 1.0, 1.0], [0.0, 0.0, -0.5]),
        ("_top", [0, 1, 2], [1.0, 1.0, 1.0], [0.0, 0.0, 0.5]),
    ]

    def __init__(
        self,
        composite_mesh_name: str,
        patch_list: Sequence[RegularMesh],
        connection_list: Sequence[PatchManager.PatchConnections],
        global_bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Creates a new regular composite mesh from a sequence of `RegularMesh` patches.

        The constructor signature matches `CompositeMesh.__init__`; the only
        difference is the narrower type for `patch_list`. Like `CompositeMesh`, this
        is not intended to be called directly: use `RegularCompositeMeshFactory` or
        the `cubed_sphere` classmethod instead.
        """
        super().__init__(
            composite_mesh_name=composite_mesh_name,
            patch_list=patch_list,
            connection_list=connection_list,
            global_bsdf=global_bsdf,
        )

    @classmethod
    def from_file(cls, path: Path) -> "RegularCompositeMesh":
        """
        Loads a `RegularCompositeMesh` from a jsonpickle file.

        `path` is a JSON file previously written with `jsonpickle.encode` (e.g. by
        `RegularCompositeMeshOptimizer.save_callback`). This is the mesh-side analog
        of `TessellatedSplineSurface.from_file`. BSDF / texture information is not
        preserved: the patches and global mesh get default BSDFs. A Mitsuba
        variant must already be set by the caller.

        :returns: The decoded `RegularCompositeMesh`.
        :raises FileNotFoundError: If `path` does not exist.
        :raises ValueError: If `path` is not a file or does not decode to a
            `RegularCompositeMesh`.
        """
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file.")

        with open(path, "r") as composite_file:
            composite_mesh = jsonpickle.decode(composite_file.read())

        if not isinstance(composite_mesh, RegularCompositeMesh):
            raise ValueError(f"File {path} does not contain a RegularCompositeMesh.")
        return composite_mesh

    def h_refinement(self, factor: int = 2) -> None:
        """
        Refines every patch by `factor` along both grid directions and rebuilds the
        global representation.

        After refinement, each patch has `factor ** 2` times as many faces and
        approximately `factor ** 2` times as many vertices (modulo shared-seam
        deduplication, which is handled by the rebuilt patch-to-global mappings).
        Vertex normals are re-averaged at the shared seams via
        `update_global_representation`.

        :raises ValueError: If `factor` is not an integer greater than 1.
        """
        if factor <= 1:
            raise ValueError("Refinement factor must be greater than 1.")

        for patch in self._patch_list:
            patch.h_refinement(factor=factor)

        self._build_mappings()
        self._averaging_weights = self._build_averaging_weights()

        self._build_global_mesh(self._global_mesh.bsdf)
        self.update_global_representation()

    def transform_global_vertices(self, transform: "mi.Transform4f") -> None:
        """
        Applies a 4x4 transformation matrix `transform` to all global vertices and
        propagates the change to every patch.

        Mesh-side analog of
        `TessellatedSplineSurface.transform_global_control_points` at
        `rebel_shape/meshing/tessellated_spline_surface.py:281`. Vertex normals
        are re-averaged at the patch seams via `update_global_representation`
        after the new positions are propagated, so they end up consistent across
        shared edges.
        """
        original_vertex_points = dr.unravel(
            dtype=mi.Point3f,
            array=self.global_vertices.array,
        )

        new_vertex_points = transform @ original_vertex_points
        new_vertices = dr.reshape(
            dtype=mi.TensorXf,
            value=dr.ravel(new_vertex_points),
            shape=self.global_vertices.shape,
        )

        self.global_vertices = new_vertices
        self.update_global_representation()

    @staticmethod
    def _transform_patch_to_cube_face(
        patch: RegularMesh,
        axis_permutation: list[int],
        scale: list[float],
        translation: list[float],
    ) -> None:
        """
        Applies an axis permutation, anisotropic scale, and translation to the
        vertices of a unit-square patch so it lands on its cube face.

        Operates in-place: the patch's vertex buffer is overwritten and Mitsuba's
        normal recomputation runs automatically (vertices setter calls
        `_mesh_params.update()`). Texture coordinates are unchanged: they stay in
        the local `[0, 1]^2` patch frame, which is what `PatchManager` uses to
        determine side membership during linking.
        """
        v = patch.vertices.numpy().astype(np.float64)
        permuted = (np.eye(3)[axis_permutation] @ v.T).T
        v_transformed = permuted * np.array([scale]) + np.array([translation])
        patch.vertices = mi.TensorXf(v_transformed)

    @classmethod
    def cubed_sphere(
        cls,
        name: str,
        subdivisions: int = 8,
        radius: float = 1.0,
        bsdf: DiffuseBSDF | None = None,
        global_bsdf: DiffuseBSDF | None = None,
    ) -> "RegularCompositeMesh":
        """
        Builds a 6-patch cubed-sphere `RegularCompositeMesh` of given `radius`.

        Each face of the unit cube is a `RegularMesh.unit_square` with
        `subdivisions` cells per side. Faces are placed in their cube positions
        using the same axis permutations as the spline `unit_cube` (see
        `rebel_shape/splines/cube_initialization.py:145-176`) so the resulting
        topology, and the `link_sides` connectivity, matches the spline cubed
        sphere exactly. After building the composite, every global vertex is
        projected radially onto the sphere of given `radius`.

        The same `bsdf` is attached to all 6 patches; Mitsuba clones the texture
        data on scene load so each patch ends up with its own optimizable
        per-patch texture buffer (this is the mesh-side analog of how
        `cubed_sphere` from the spline path produces 6 independent textures).

        :raises ValueError: If `name` is empty, `subdivisions < 1`, or `radius <= 0`.
        """
        if len(name) == 0:
            raise ValueError("`name` must be a non-empty string.")
        if subdivisions < 1:
            raise ValueError("`subdivisions` must be at least 1.")
        if radius <= 0.0:
            raise ValueError("`radius` must be a positive real number.")

        if bsdf is None:
            bsdf = DiffuseBSDF()

        # Avoid a forward-reference cycle: the factory imports this class.
        from rebel_shape.meshing.regular_composite_mesh_factory import (
            RegularCompositeMeshFactory,
        )

        patches_by_face: dict[str, RegularMesh] = {}
        for suffix, perm, scale, translation in cls._CUBE_FACE_TRANSFORMS:
            patch = RegularMesh.unit_square(
                mesh_name=name + suffix,
                width_subdivisions=subdivisions,
                height_subdivisions=subdivisions,
                bsdf=bsdf,
            )
            cls._transform_patch_to_cube_face(
                patch=patch,
                axis_permutation=perm,
                scale=scale,
                translation=translation,
            )
            patches_by_face[suffix] = patch

        left = patches_by_face["_left"]
        right = patches_by_face["_right"]
        back = patches_by_face["_back"]
        front = patches_by_face["_front"]
        bottom = patches_by_face["_bottom"]
        top = patches_by_face["_top"]

        factory = RegularCompositeMeshFactory(manager_name=name, bsdf=global_bsdf)
        for patch in (left, right, back, front, bottom, top):
            factory.add_patch(patch)

        # Top face links its four sides to the four side faces' TOP sides
        for second_patch, side in [
            (left, Patch.Side.LEFT),
            (right, Patch.Side.RIGHT),
            (front, Patch.Side.TOP),
            (back, Patch.Side.BOTTOM),
        ]:
            factory.link_sides(
                first_patch_name=top.name,
                first_patch_side=side,
                second_patch_name=second_patch.name,
                second_patch_side=Patch.Side.TOP,
            )
        # Bottom face links its four sides to the four side faces' BOTTOM sides
        for second_patch, side in [
            (left, Patch.Side.RIGHT),
            (right, Patch.Side.LEFT),
            (front, Patch.Side.TOP),
            (back, Patch.Side.BOTTOM),
        ]:
            factory.link_sides(
                first_patch_name=bottom.name,
                first_patch_side=side,
                second_patch_name=second_patch.name,
                second_patch_side=Patch.Side.BOTTOM,
            )
        # The four side faces form a horizontal ring
        for first_patch, second_patch in [
            (right, front),
            (front, left),
            (left, back),
            (back, right),
        ]:
            factory.link_sides(
                first_patch_name=first_patch.name,
                first_patch_side=Patch.Side.RIGHT,
                second_patch_name=second_patch.name,
                second_patch_side=Patch.Side.LEFT,
            )

        composite_mesh = factory.build()

        # Project all global vertices onto the sphere of given radius
        global_vertices_np = composite_mesh.global_vertices.numpy()
        magnitudes = np.linalg.norm(global_vertices_np, axis=1, keepdims=True)
        composite_mesh.global_vertices = mi.TensorXf(
            global_vertices_np / magnitudes * radius
        )

        # The patches were built flat, so their normals are the cube's face
        # normals, and moving the vertices does not refresh them. Derive them
        # from the projected geometry, then average across the seams so shared
        # vertices agree. They are deliberately not set to the radial direction:
        # this mesh is a sphere only at initialization and is deformed from
        # there on, so the normals have to follow the actual faces.
        for patch in composite_mesh:
            patch.recompute_vertex_normals()
        composite_mesh.update_global_representation()

        return composite_mesh
