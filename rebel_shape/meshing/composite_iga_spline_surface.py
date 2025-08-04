from collections.abc import Iterator, Sequence
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from rebel_shape.meshing.composite_spline_surface import CompositeSplineSurface
from rebel_shape.meshing.patch_manager import PatchManager
from rebel_shape.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class CompositeIGASplineSurface(CompositeSplineSurface):
    """
    A class that handles multiple `IGASplinePatch` objects and manages their connections
    to form a composite spline surface for isogeometric analysis (IGA).

    It extends the `CompositeSplineSurface` with global field values used for IGA
    applications, where each patch has its own field values alongside global field
    values which are maintained to ensure continuity across patches.

    This class works with `IGASplinePatch` objects which are a wrapper around
    concrete classes implementing the `SplinePatch` interface. They can thus be used
    in the same manner as `SplinePatch` objects and passed to the underlying
    `CompositeSplineSurface` methods.

    Typically, this class is used after the spline based shape optimization has been
    executed, and one wants to run isogeometric analysis on the optimized
    `CompositeSplineSurface`.
    """

    _patch_list: Sequence[IGASplinePatch]
    """Array of `IGASplinePatch` instances that constitute the composite surface"""
    _global_field_values: "mi.TensorXf"
    """
    Global field values of the composite spline surface, which are a tensor of shape
    (n_global_control_points, n_dofs_per_cp). Ensures continuity across patches. 
    """

    def __init__(
        self,
        composite_spline_name: str,
        patch_list: Sequence[IGASplinePatch],
        connection_list: Sequence[PatchManager.PatchConnections],
        bsdf: DiffuseBSDF | None = None,
    ) -> None:
        """
        Creates a new composite IGA spline surface with unique name
        `composite_spline_name`.

        The composite spline surface is built from a list of `IGASplinePatch` objects
        (`patch_list`) and their connections (`connection_list`).
        The `bsdf` parameter is optional and can be used to specify a BSDF for the
        global mesh created by `CompositeMesh`.

        :raises ValueError: If the number of patches and connection lists do not
            match, or if any of the patches does not have the same number of
            degrees of freedom per control point as the first patch in the list.
        """
        reference_dofs_per_cp = patch_list[0].field_values.shape[1]
        for patch in patch_list:
            if patch.field_values.shape[1] != reference_dofs_per_cp:
                raise ValueError(
                    "All patches must have the same number of degrees of freedom per "
                    "control point."
                )

        self._dofs_per_cp: int = reference_dofs_per_cp

        super().__init__(
            composite_spline_name=composite_spline_name,
            patch_list=patch_list,
            connection_list=connection_list,
            bsdf=bsdf,
        )

    @classmethod
    def from_spline_surface(
        cls,
        spline_surface: CompositeSplineSurface,
        per_patch_field_values: list["mi.TensorXf"],
    ) -> "CompositeIGASplineSurface":
        """
        Creates a `CompositeIGASplineSurface` from an existing `CompositeSplineSurface`
        and a list of field values for each patch.

        This method is useful when the optimized spline surface is already
        represented as a `CompositeSplineSurface`, but one wants to convert it
        to an IGA-compatible representation.

        :returns: An instance of `CompositeIGASplineSurface`.
        """
        if len(spline_surface._patch_list) != len(per_patch_field_values):
            raise ValueError(
                "The number of patches in the spline surface must match the number "
                "of provided field values."
            )

        iga_patch_list = [
            IGASplinePatch(spline_patch=patch, field_values=field_values)
            for patch, field_values in zip(
                spline_surface._patch_list, per_patch_field_values
            )
        ]

        return cls(
            composite_spline_name=spline_surface.name,
            patch_list=iga_patch_list,
            connection_list=spline_surface._connection_list,
            bsdf=spline_surface._bsdf,
        )

    def _build_global_representation(self) -> None:
        """
        Builds the global field values as well as the global mesh (parent method).

        Unlike building the mapping, this method is less expensive as it can be
        vectorized, but it still requires to go through all the patches to
        construct the global control points.

        The field values of the control points on the edges are not averaged but
        rather the last patch that was inserted is used to determine the value of the
        field. This choice is not propagated to the patches so the representation is
        not fully consistent yet. Calling `update_global_representation` will average
        the field values of the control points across all patches and propagate the
        changes to all connected patches.

        Global mesh building is handled by the parent class method.
        """
        super()._build_global_representation()

        global_values: npt.NDArray[np.float32] = np.zeros(
            shape=(len(self._global_to_patches_map), self._dofs_per_cp),
            dtype=np.float32,
        )

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = self._patch_to_global_map[patch_idx]
            global_values[local_to_global] = patch.field_values.numpy()

        self._global_field_values = mi.TensorXf(global_values)

    def update_global_representation(self) -> None:
        """
        Updates the global control points and field values of the composite spline
        surface.

        To avoid discontinuities between patches, the control points of each patch
        are averaged based on the global control points mapping. The averaged
        positions are then propagated to the local control points of each patch.

        Same applies to the field values of each patch; executing IGA codes requires
        continuous field values across patches.
        """
        # superclass updates the global control points
        super().update_global_representation()

        new_values = dr.zeros(dtype=mi.TensorXf, shape=self._global_field_values.shape)
        assert new_values.shape[0] == len(self._global_to_patches_map)

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = mi.TensorXu(self._patch_to_global_map[patch_idx])

            self._scatter_values(
                target=new_values, values=patch.field_values, indices=local_to_global
            )

        self.global_field_values = self._averaging_weights * new_values

    def __getstate__(self) -> dict[str, Any]:
        """
        Exports the composite IGA spline surface state for serialization.

        On top of the parent class state, it includes `_global_field_values`
        which is a Mitsuba tensor of shape (n_global_control_points, n_dofs_per_cp)
        that has to be converted to a numpy array for serialization.

        :returns: A dictionary representing the state of the object, ready for
                  serialization with `jsonpickle`.
        """
        state = super().__getstate__()
        state["_global_field_values"] = self._global_field_values.numpy()

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore the composite spline surface state from a serialized state.

        `state` is a dictionary that contains the serialized state of the object
        including `_global_field_values`, which is a numpy array that needs to be
        converted back to a Mitsuba tensor. The parent updates the remaining state.
        """
        # reconstruct foo from the array
        global_field_values = state.pop("_global_field_values")

        super().__setstate__(state)
        self._global_field_values = mi.TensorXf(global_field_values)

    def __getitem__(self, item: int) -> IGASplinePatch:
        """
        Returns the spline patch at the specified index.

        :raise IndexError: If the index is out of range.
        """
        if item < 0 or item >= len(self._patch_list):
            raise IndexError("Mesh index out of range.")

        return self._patch_list[item]

    def __iter__(self) -> Iterator[IGASplinePatch]:
        """
        Returns an iterator over the IGA patches in the composite IGA spline surface.
        """
        return iter(self._patch_list)

    def patches_with_mapping(
        self,
    ) -> list[tuple[npt.NDArray[np.uint32], IGASplinePatch]]:
        """
        Returns a list of tuples containing the local-to-global mapping and the
        corresponding `IGASplinePatch` for each patch in the composite spline surface.
        """
        return [
            (self._patch_to_global_map[i], patch)
            for i, patch in enumerate(self._patch_list)
        ]

    @property
    def global_field_values(self) -> "mi.TensorXf":
        """
        Returns the global field values of the composite B-spline surface.

        The field values are stored in a Mitsuba tensor of shape (n_cp, n_dofs_per_cp),
        where n_cp is the number of control points and n_dofs_per_cp is the number
        of degrees of freedom per control point.
        """
        return self._global_field_values

    @global_field_values.setter
    def global_field_values(self, field_values: "mi.TensorXf") -> None:
        """
        Sets the global field values of the composite B-spline surface and propagates
        the changes to the local field values of each patch.

        The field values are a Mitsuba tensor of shape (n_cp, n_dofs_per_cp), where
        n_cp is the number of control points and n_dofs_per_cp is the number
        of degrees of freedom per control point.

        :raises ValueError: If the shape of the provided field values does not match
            the shape of the current global field values.
        """
        if self._global_field_values.shape != field_values.shape:
            raise ValueError(
                f"Expected field values shape {self._global_field_values.shape}, "
                f"but got {field_values.shape}."
            )

        self._global_field_values = field_values

        for patch_idx, patch in enumerate(self._patch_list):
            local_to_global = dr.ravel(
                mi.TensorXf(self._patch_to_global_map[patch_idx])
            )
            patch.field_values = self._global_field_values[local_to_global]

    @property
    def degree(self):
        """
        Returns the degree of the spline surface.

        The degree is determined by finding the maximum degree among all patches in
        the composite spline surface.
        """
        max_degree = 0
        for patch in self._patch_list:
            max_degree = max(max_degree, patch.degree)

        return max_degree
