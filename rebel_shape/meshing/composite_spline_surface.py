from collections.abc import Iterator, Sequence
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCP.TopoDS import TopoDS_Shape

from rebel_shape.meshing.composite_mesh import CompositeMesh
from rebel_shape.meshing.patch_manager import PatchManager
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.splines.spline_patch import SplinePatch


class CompositeSplineSurface(PatchManager):
    """
    A class that handles multiple `SplinePatch` objects and manages their connections
    to form a composite spline surface.

    Unlike `CompositeMesh`, this class does not create a global spline as this is not
    a well-defined operation for splines - every patch has its own parametric domain.
    However, control points are shared between patches, necessitating a global
    control point mapping to avoid discontinuities.

    This class works with any type of `SplinePatch`, but at the time of writing,
    only the `BSplinePatch` is implemented. Support for other tensor product
    patches (like NURBS) is straightforward, while more complex spline patches like
    T-splines will require some additional work to ensure proper connectivity.

    Spline patches cannot be optimized directly, so a method is provided to
    create a `CompositeMesh` from the tessellated patches of the composite spline
    mesh.
    """

    _patch_list: Sequence[SplinePatch]
    """Array of `SplinePatch` instances that constitute the composite surface"""
    _global_control_points: "mi.TensorXf"
    """
    Global control points of the composite spline surface, shaped as (n, 3) where n 
    is the number of control points.
    """
    _global_quads: "mi.TensorXu"
    """
    Global quads which contain the indices of the quads that make up the
    global spline surface.
    """

    def __init__(
        self,
        composite_spline_name: str,
        patch_list: Sequence[SplinePatch],
        connection_list: Sequence[PatchManager.PatchConnections],
    ) -> None:
        """
        Creates a new composite spline surface with unique name `composite_spline_name`.

        The composite spline surface is built from a list of `SplinePatch` objects
        (`patch_list`) and their connections (`connection_list`).

        :raises ValueError: If the number of patches and connection lists do not match,
            or if they are empty.
        """
        super().__init__(
            name=composite_spline_name,
            patch_list=patch_list,
            connection_list=connection_list,
        )

        self._build_global_representation()
        self.update_global_representation()

    def _build_global_representation(self) -> None:
        """
        Builds the global mesh control points from the patches.

        Unlike building the mapping, this method is less expensive as it can be
        vectorized, but it still requires to go through all the patches to
        construct the global control points.

        The position of the control points on the edges is not averaged but rather the
        last patch that was inserted is used to determine the position of the
        control point. This choice is not propagated to the patches so the
        representation is not fully consistent yet. Calling
        `update_global_representation` will average the control points across all
        patches and propagate the changes to all connected patches.
        """
        global_cps: npt.NDArray[np.float32] = np.zeros(
            shape=(len(self._global_to_patches_map), 3), dtype=np.float32
        )
        global_quads: list[npt.NDArray[np.uint32]] = []

        for patch, mapping in self.patches_with_mapping:
            global_cps[mapping] = patch.control_points.numpy()
            global_quads.extend(mapping[patch.quads.numpy()])

        self._global_control_points = mi.TensorXf(global_cps)
        self._global_quads = mi.TensorXu(np.array(global_quads, dtype=np.uint32))

    def __getstate__(self) -> dict[str, Any]:
        """
        Exports the composite spline surface state for serialization.

        `_global_control_points`, `_global_quads`, and `_averaging_weights` are
        not serializable, so we need to convert them to numpy arrays before
        serializing the object state.

        :returns: A dictionary representing the state of the object, ready for
                  serialization with `jsonpickle`.
        """
        state = self.__dict__.copy()

        state["_averaging_weights"] = self._averaging_weights.numpy()
        state["_global_control_points"] = self._global_control_points.numpy()
        state["_global_quads"] = self._global_quads.numpy()

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore the composite spline surface state from a serialized state.

        `state` is a dictionary that contains the serialized state of the object
        including values `_global_control_points`, `_global_quads`, and
        `_averaging_weights` which converted from numpy arrays to Mitsuba tensors.
        """
        # reconstruct foo from the array
        global_control_points = state.pop("_global_control_points")
        global_quads = state.pop("_global_quads")
        averaging_weights = state.pop("_averaging_weights")

        # For compatibility with legacy serialized objects that still contain this
        # attribute, we remove it if it exists.
        if "_bsdf" in state:
            _ = state.pop("_bsdf")

        self.__dict__.update(state)

        self._global_control_points = mi.TensorXf(global_control_points)
        self._global_quads = mi.TensorXu(global_quads)
        self._averaging_weights = mi.TensorXf(averaging_weights)

    def update_global_representation(self) -> None:
        """
        Updates the global control points of the composite spline surface.

        To avoid discontinuities between patches, the control points of each patch
        are averaged based on the global control points mapping. The averaged
        positions are then propagated to the local control points of each patch.
        """
        new_cps = dr.zeros(dtype=mi.TensorXf, shape=self.global_control_points.shape)
        assert new_cps.shape[0] == len(self._global_to_patches_map)

        for patch, mapping in self.patches_with_mapping:
            self._scatter_values(
                target=new_cps,
                values=patch.control_points,
                indices=mi.TensorXu(mapping),
            )

        self.global_control_points = self._averaging_weights * new_cps

    def __getitem__(self, item: int) -> SplinePatch:
        """
        Returns the spline patch at the specified index.

        :raise IndexError: If the index is out of range.
        """
        if item < 0 or item >= len(self._patch_list):
            raise IndexError("Mesh index out of range.")

        return self._patch_list[item]

    def __iter__(self) -> Iterator[SplinePatch]:
        """
        Returns an iterator over the patches in the composite spline surface.
        """
        return iter(self._patch_list)

    def h_refinement(self, factor: int = 2) -> None:
        """
        Refines the composite spline surface in the horizontal direction by a factor.

        This method refines each patch in the composite spline surface by the specified
        factor, effectively increasing the resolution of the surface in the horizontal
        direction.

        After refinement, the global control points, quads, and averaging weights have
        to be rebuilt to reflect the extensive changes in the patches.

        :raises ValueError: If the factor is less than or equal to 1.
        """
        if factor <= 1:
            raise ValueError("Refinement factor must be greater than 1.")

        for patch in self._patch_list:
            patch.h_refinement(factor=factor)

        self._build_mappings()
        self._averaging_weights = self._build_averaging_weights()

        self._build_global_representation()
        self.update_global_representation()

    @property
    def global_control_points(self) -> "mi.TensorXf":
        """
        Returns the global control points of the composite B-spline surface.

        The control points are stored in a Mitsuba tensor of shape (n, 3), where n is
        the number of control points.
        """
        return self._global_control_points

    @global_control_points.setter
    def global_control_points(self, control_points: "mi.TensorXf") -> None:
        """
        Sets the global control points of the composite B-spline surface and propagates
        the changes to the local control points of each patch.

        The control points are expected to be a Mitsuba tensor of shape (n, 3), where n
        is the number of control points.

        :raises ValueError: If the shape of the provided control points does not match
            the shape of the current global control points.
        """
        if self._global_control_points.shape != control_points.shape:
            raise ValueError(
                f"Expected control points shape {self._global_control_points.shape}, "
                f"but got {control_points.shape}."
            )

        self._global_control_points = control_points

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = dr.ravel(
                mi.TensorXf(self._patch_to_global_map[patch_idx])
            )
            patch.control_points = self._global_control_points[local_to_global]

    @property
    def global_quads(self):
        """
        Global quads which contain the indices of the quads that make up the
        composite spline surface.

        The quads are stored in a Mitsuba tensor of shape (n_quads, 4), where n_quads
        is the number of quads in the composite spline surface.
        """
        return self._global_quads

    @property
    def degree(self):
        """
        Returns the degree of the spline surface.

        All patches must have the same degree. Raises `ValueError` if
        patches have different degrees.
        """
        degrees = {patch.degree for patch in self._patch_list}
        if len(degrees) > 1:
            raise ValueError(
                f"Composite surface has mixed patch degrees: {degrees}. "
                "All patches must have the same degree."
            )
        return degrees.pop()

    @property
    def patches_with_mapping(
        self,
    ) -> Iterator[tuple[SplinePatch, npt.NDArray[np.uint32]]]:
        """
        Returns an iterator of tuples containing the spline patches with the
        corresponding local-to-global mapping.
        """
        return zip(self._patch_list, self._patch_to_global_map)

    def composite_mesh_tesselation(
        self,
        subdivisions_per_patch: int = 32,
        per_patch_bsdf: Sequence[DiffuseBSDF] | None = None,
        global_mesh_bsdf: DiffuseBSDF | None = None,
    ) -> CompositeMesh:
        """
        Creates a `CompositeMesh` from tessellated spline patches of the composite
        spline surface.

        This method is meant to bridge the gap between the spline surface and a
        triangulated representation suitable for rendering or further processing.
        The `subdivisions_per_patch` parameter specifies the number of subdivisions
        in the u and v directions for each patch.

        Since the connectivity information is already stored in this class (it
        inherits from `PatchManager`), the constructor for `CompositeMesh` can
        be used directly to create a composite mesh from the tessellated patches,
        bypassing the need for a factory method. This is an exception to the rule!

        The tessellated mesh has the same name as the composite spline surface with
        "_tessellated" appended to it.

        `per_patch_bsdf` is an optional sequence of BSDFs that are assigned to each
        patch mesh in the tessellated surface. It must match the number of patches
        in the composite spline surface and be ordered in the same way as the
        patches. If not provided, the patches will receive a default DiffuseBSDF.

        The `global_mesh_bsdf` is an optional BSDF that is assigned to the
        global tessellated mesh (union of all patch meshes). If not provided,
        the global mesh will receive a default DiffuseBSDF.


        :returns: A `CompositeMesh` built from the tessellated patches of the
                  composite spline surface.

        :raises ValueError: If `subdivisions_per_patch` is less than 1 or if the
            number of BSDFs provided does not match the number of patches.
        """
        if subdivisions_per_patch < 1:
            raise ValueError("The number of subdivisions per patch must be at least 1.")

        if per_patch_bsdf is not None and len(per_patch_bsdf) != len(self._patch_list):
            raise ValueError(
                "The number of BSDFs provided must match the number of patches."
            )

        if per_patch_bsdf is None:
            per_patch_bsdf = [DiffuseBSDF()] * len(self._patch_list)

        tessellated_patches = [
            patch.tessellate(
                u_subdivisions=subdivisions_per_patch,
                v_subdivisions=subdivisions_per_patch,
                bsdf=bsdf,
            )
            for patch, bsdf in zip(self._patch_list, per_patch_bsdf)
        ]

        composite_mesh = CompositeMesh(
            composite_mesh_name=self.name + "_tessellated",
            patch_list=tessellated_patches,
            connection_list=self._connection_list,
            global_bsdf=global_mesh_bsdf,
        )

        return composite_mesh

    def to_brep(self, tolerance: float = 1e-6) -> TopoDS_Shape:
        """
        Export the composite spline surface as a stitched B-Rep quad model.

        This method creates individual OCP B-Spline surfaces from each patch using
        the `to_brep` method, converts them to faces, and then stitches them
        together using OCP's sewing algorithm. The result is a single TopoDS_Shape
        that represents the entire composite surface as a B-Rep model.

        The stitching process attempts to merge edges that are within the specified
        `tolerance` distance of each other, creating a watertight B-Rep model if
        the patches are properly connected. Due to the global control point
        synchronization enforced by the class, this parameter is a CAD formality.
        Smallest OCCT absolute accuracy is 1e-7 (the library does not have relative
        accuracies).

        :returns: A TopoDS_Shape representing the stitched B-Rep model of the
                 composite spline surface.

        :raises ValueError: If the stitching process fails for any patch.
        """
        sewing = BRepBuilderAPI_Sewing(tolerance)

        # Convert each patch to a B-Rep face and add to the sewing operation
        for patch in self._patch_list:
            bspline_surface = patch.to_brep()

            # The face is created over the full parametric domain of the surface
            face_maker = BRepBuilderAPI_MakeFace(bspline_surface, tolerance)

            if not face_maker.IsDone():
                raise RuntimeError(
                    f"Failed to create B-Rep face from patch {patch.name}"
                )

            face = face_maker.Face()
            sewing.Add(face)

        sewing.Perform()

        return sewing.SewedShape()
