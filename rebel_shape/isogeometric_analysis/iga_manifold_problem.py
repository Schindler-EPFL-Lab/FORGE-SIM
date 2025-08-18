import drjit as dr
import mitsuba as mi
import numpy as np

from rebel_shape.isogeometric_analysis.iga_problem import IGAProblem
from rebel_shape.meshing.composite_iga_spline_surface import CompositeIGASplineSurface
from rebel_shape.splines.iga_spline_patch import IGASplinePatch


class IGAManifoldProblem(IGAProblem):
    """
    Specialization of the `IGAProblem` class for solving IGA problems on 2D manifolds.

    This abstract base class provides common functionality for isogeometric analysis
    problems defined on 2D manifolds embedded in 3D space. It extracts recurring
    patterns that emerged during the implementation of concrete manifold problems,
    particularly operations related to assembling and combining local matrices and
    vectors from multiple patches into unified global structures.

    It provides utility methods for:

    - Concatenating local matrices and vectors from multiple patches
    - Transforming basis function derivatives using metric tensors
    - Generating scatter indices for local matrix assembly

    It also includes a `KnotSpanHandler` class for managing knot spans in IGA spline
    patches, which provides methods for computing midpoints of knot spans and performing
    change of interval operations for Gauss-Legendre quadrature.
    """

    class KnotSpanHandler:
        """
        A utility class for handling knot spans in IGA manifold problems.

        This class encapsulates the management of knot spans for isogeometric analysis
        on 2D manifolds, providing essential functionality for numerical integration
        using Gauss-Legendre quadrature. It handles the transformation of quadrature
        points and weights from the standard [-1, 1] interval to the actual knot span
        intervals in the parametric domain.

        The class pre-computes knot span properties such as lengths and midpoints to
        enable efficient quadrature rule application across all knot spans (vectorized).

        Key capabilities include:
        - Transformation of Gauss-Legendre quadrature points to knot span coordinates
        - Scaling of quadrature weights for proper integration over knot spans
        - Efficient access to knot span geometric properties
        - Generation of complete quadrature rules for 2D integration
        """

        def __init__(
            self, patch: IGASplinePatch, integration_order: int | None = None
        ) -> None:
            """
            Initializes the KnotSpanHandler with a given `patch`.

            This constructor sets up the knot span handler by extracting knot span
            information from the provided spline patch and preparing data structures
            for efficient quadrature operations. It precomputes knot span lengths,
            midpoints, and Gauss-Legendre quadrature points and weights.

            The `integration_order` determines the accuracy of numerical integration.
            Higher orders provide more accurate integration but at increased
            computational cost. The default integration order equals the spline degree,
            which is sufficient for many applications.

            :raises ValueError: If integration_order is less than 1.
            """
            if integration_order is None:
                integration_order = patch.interpolating_spline.degree

            if integration_order < 1:
                raise ValueError(
                    f"Integration order must be at least 1, got {integration_order}."
                )
            self._integration_order = integration_order

            self._knot_spans = patch.interpolating_spline.knot_spans

            self._lengths_u = self._knot_spans[:, 0, 1] - self._knot_spans[:, 0, 0]
            self._lengths_v = self._knot_spans[:, 1, 1] - self._knot_spans[:, 1, 0]

            self._midpoint_u = 0.5 * (
                self._knot_spans[:, 0, 0] + self._knot_spans[:, 0, 1]
            )
            self._midpoint_v = 0.5 * (
                self._knot_spans[:, 1, 0] + self._knot_spans[:, 1, 1]
            )

            self._gl_positions, self._gl_weights = np.polynomial.legendre.leggauss(
                deg=self._integration_order
            )

        @property
        def midpoint_parametric_coordinates(self) -> "mi.TensorXf":
            """
            Returns the midpoints of the knot spans in parametric coordinates.

            This property provides the center points of all knot spans in the parametric
            domain (u, v). The midpoints are useful for getting the active basis
            functions in a particular knot span and for very approximate evaluation
            of the spline surface at the center of each knot span.

            Midpoints are computed for all knot spans (vectorized) as follows:
                midpoint_u = 0.5 * (u_start + u_end)
                midpoint_v = 0.5 * (v_start + v_end)

            where u_start, u_end, v_start, v_end are the boundaries of each knot span.

            :returns: A tensor of shape (n_knot_spans, 2) with midpoints in u and v
                directions. The first column contains u-coordinates and the second
                column contains v-coordinates.
            """
            parametric_coords = dr.empty(
                dtype=mi.TensorXf, shape=(self.num_knot_spans, 2)
            )
            parametric_coords[:, 0] = dr.ravel(self._midpoint_u)
            parametric_coords[:, 1] = dr.ravel(self._midpoint_v)

            return parametric_coords

        @property
        def num_knot_spans(self) -> int:
            """
            Returns the number of knot spans in the patch.

            This property provides the total number of knot spans in the spline patch,
            which corresponds to the number of elements in the isogeometric
            discretization. Each knot span represents a region where the spline
            basis functions have local support.

            :returns: The number of knot spans in the patch.
            """
            return self._knot_spans.shape[0]

        def _gl_interval_change(
            self, sample_u: float, sample_v: float
        ) -> "mi.TensorXf":
            """
            Performs a change of interval for input offsets in u and v directions.

            This method transforms Gauss-Legendre quadrature points from the standard
            [-1, 1] interval to the actual knot span intervals in the parametric domain.
            Since Gauss-Legendre quadrature is defined on [-1, 1], this transformation
            is essential for proper numerical integration over arbitrary knot spans.

            The transformation is performed using the linear mapping:
                u_actual = midpoint_u + 0.5 * length_u * `sample_u`
                v_actual = midpoint_v + 0.5 * length_v * `sample_v`

            where length_u and length_v are the lengths of each knot span in the
            respective directions.

            :returns: A tensor of shape (n_knot_spans, 2) containing the transformed
                parametric coordinates. The first column contains u-coordinates and
                the second column contains v-coordinates.
            """
            parametric_coords = dr.empty(
                dtype=mi.TensorXf, shape=(self.num_knot_spans, 2)
            )
            parametric_coords[:, 0] = dr.ravel(
                self._midpoint_u + 0.5 * self._lengths_u * sample_u
            )
            parametric_coords[:, 1] = dr.ravel(
                self._midpoint_v + 0.5 * self._lengths_v * sample_v
            )

            return parametric_coords

        def _gl_weight_scaling(self, weight_u: float, weight_v: float) -> "mi.TensorXf":
            """
            Performs a change of interval for Gauss-Legendre quadrature weights.

            This method scales the Gauss-Legendre quadrature weights from the standard
            [-1, 1] interval to account for the actual lengths of the knot spans in
            the parametric domain. The scaling ensures that numerical integration
            produces correct results over the transformed intervals.

            The weight scaling follows the Jacobian of the interval transformation:
                scaled_weight = 0.25 * length_u * length_v * weight_u * weight_v

            where the factor 0.25 comes from the Jacobian of the transformation from
            [-1, 1] × [-1, 1] to [0, length_u] × [0, length_v], and the lengths
            account for the actual knot span dimensions.

            :returns: A tensor of shape (n_knot_spans,) containing the scaled weights
                for each knot span.
            """
            return 0.25 * (self._lengths_u * weight_u) * (self._lengths_v * weight_v)

        def gl_coords_and_weights(self) -> list[tuple["mi.TensorXf", "mi.TensorXf"]]:
            """
            Returns a list of tuples containing evaluation coordinates and weights
            for Gauss-Legendre quadrature for each sample point.

            This method generates a complete set of quadrature points and weights for
            2D numerical integration over all knot spans. It combines the 1D
            Gauss-Legendre rules in both parametric directions to create a tensor
            product quadrature rule suitable for integration over rectangular knot
            spans.

            :returns: A list of tuples where each tuple contains:

                - parametric_coords: A tensor of shape (n_knot_spans, 2) with the
                  transformed parametric coordinates for a specific quadrature point.
                - scaled_weight: A tensor of shape (n_knot_spans,) with the scaled
                  quadrature weights for the corresponding coordinates.

                The list length equals `integration_order`^2, representing all
                combinations of 1D quadrature points in the 2D tensor product rule.
            """
            evaluation_coords_and_weights: list[
                tuple["mi.TensorXf", "mi.TensorXf"]
            ] = []

            for pos_v, weight_v in zip(self._gl_positions, self._gl_weights):
                for pos_u, weight_u in zip(self._gl_positions, self._gl_weights):
                    parametric_coords = self._gl_interval_change(
                        sample_u=pos_u, sample_v=pos_v
                    )
                    scaled_weight = self._gl_weight_scaling(
                        weight_u=weight_u, weight_v=weight_v
                    )

                    evaluation_coords_and_weights.append(
                        (parametric_coords, scaled_weight)
                    )

            return evaluation_coords_and_weights

        @property
        def min_span_size(self) -> float:
            """
            Returns the minimum size of the knot spans in the patch.

            This property computes the smallest knot span dimension across both
            parametric directions (u and v) in the entire patch. The minimum span
            size is useful for determining appropriate regularization parameters,
            time step sizes, and convergence criteria in numerical algorithms.

            :returns: The minimum size of the knot spans as a positive float value.
            """
            return float(min(dr.min(self._lengths_u), dr.min(self._lengths_v)).numpy())

        @property
        def max_span_size(self) -> float:
            """
            Returns the maximum size of the knot spans in the patch.

            This property computes the largest knot span dimension across both
            parametric directions (u and v) in the entire patch. The maximum span
            size is useful for determining stability conditions, error estimates,
            and adaptive refinement strategies in numerical algorithms.

            :returns: The maximum size of the knot spans as a positive float value.
            """
            return float(max(dr.max(self._lengths_u), dr.max(self._lengths_v)).numpy())

    @staticmethod
    def _concatenate_patch_local_matrices(
        composite_iga_spline: CompositeIGASplineSurface,
        local_indices_list: list["mi.TensorXu"],
        local_matrices_list: list["mi.TensorXf"],
    ) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Concatenates the local matrix tensors and local index tensors from all patches
        of a `composite_iga_spline` surface into unified tensors.

        This method receives `local_matrices_list` and their corresponding
        degree-of-freedom `local_indices_list` from multiple patches into single
        tensors that can be directly used with the `assemble_global_matrix` method of
        the `IGAProblem` class. The local indices are remapped to global indices
        using the mappings from the `composite_iga_spline` surface.

        The concatenation preserves the order of patches as they appear in the composite
        spline surface. All local matrices must have the same shape in their last two
        dimensions (number of degrees of freedom per knot span).

        :returns: A tuple containing:

            - all_local_indices: Global degree-of-freedom indices with shape
              (sum_p n_knot_spans_p, n_dofs).
            - all_local_matrices: Concatenated local matrices with shape
              (sum_p n_knot_spans_p, n_dofs, n_dofs).

        :raises ValueError: If the lengths of local_indices_list and local_matrices_list
            don't match, if the lists are empty, if the matrices from different patches
            in `local_matrices_list` have incompatible shapes, or if the indices
            from different patches in `local_indices_list` have incompatible shapes.
        """
        if len(local_indices_list) != len(local_matrices_list):
            raise ValueError(
                "Length of the local indices list and local matrices list must "
                f"be the same. Got {len(local_indices_list)} and "
                f"{len(local_matrices_list)} respectively."
            )

        if len(local_indices_list) == 0:
            raise ValueError("Cannot concatenate empty lists of local matrices.")

        # Check that all matrices have compatible shapes
        expected_dofs = local_matrices_list[0].shape[1]
        for i, matrix in enumerate(local_matrices_list):
            if matrix.shape[1] != expected_dofs or matrix.shape[2] != expected_dofs:
                raise ValueError(
                    f"All local matrices must have the same number of degrees of "
                    f"freedom. Expected shape (*, {expected_dofs}, {expected_dofs}), "
                    f"but matrix {i} has shape {matrix.shape}."
                )
        for i, indices in enumerate(local_indices_list):
            if indices.shape[1] != expected_dofs:
                raise ValueError(
                    f"All local indices must have the same number of degrees of "
                    f"freedom. Expected shape (*, {expected_dofs}), "
                    f"but indices {i} has shape {indices.shape}."
                )

        offsets: list[int] = [0]
        for local_indices in local_indices_list:
            offsets.append(offsets[-1] + local_indices.shape[0])

        all_local_matrices = dr.zeros(
            dtype=mi.TensorXf,
            shape=(
                offsets[-1],
                local_matrices_list[0].shape[1],
                local_matrices_list[0].shape[2],
            ),
        )
        all_local_indices = dr.zeros(
            dtype=mi.TensorXu, shape=(offsets[-1], local_indices_list[0].shape[1])
        )
        mappings = [mapping for _, mapping in composite_iga_spline.patches_with_mapping]

        for i, (local_indices, local_matrices, mapping) in enumerate(
            zip(local_indices_list, local_matrices_list, mappings)
        ):
            start = offsets[i]
            end = offsets[i + 1]

            all_local_matrices[start:end, :, :] = dr.ravel(local_matrices)
            all_local_indices[start:end, :] = dr.ravel(
                mi.TensorXu(mapping[local_indices])
            )

        return all_local_indices, all_local_matrices

    @staticmethod
    def _concatenate_patch_local_vectors(
        composite_iga_spline: CompositeIGASplineSurface,
        local_indices_list: list["mi.TensorXu"],
        local_vectors_list: list["mi.TensorXf"],
    ) -> tuple["mi.TensorXu", "mi.TensorXf"]:
        """
        Concatenates the local vector tensors and local index tensors from all patches
        of a composite IGA spline surface into unified tensors.

        This method receives `local_vectors_list` and their corresponding
        degree-of-freedom `local_indices_list` and combines the local vectors and
        indices from multiple patches into single tensors that can be directly used
        with the `assemble_global_vector` method of the `IGAProblem` class. The local
        indices are remapped to global indices using the patch mappings from the
        composite spline surface.

        The concatenation preserves the order of patches as they appear in the composite
        spline surface. All local vectors must have the same number of degrees of
        freedom per knot span.

        :returns: A tuple containing:

            - all_local_indices: Global degree-of-freedom indices with shape
              (sum_p n_knot_spans_p, n_dofs).
            - all_local_vectors: Concatenated local vectors with shape
              (sum_p n_knot_spans_p, n_dofs).

        :raises ValueError: If the lengths of local_indices_list and local_vectors_list
            don't match, if the lists are empty, if the vectors from different patches
            in `local_vectors_list` have incompatible shapes, or if the indices
            from different patches in `local_indices_list` have incompatible shapes.
        """
        if len(local_indices_list) != len(local_vectors_list):
            raise ValueError(
                "Length of the local indices list and local vectors list must "
                f"be the same. Got {len(local_indices_list)} and "
                f"{len(local_vectors_list)} respectively."
            )

        if len(local_indices_list) == 0:
            raise ValueError("Cannot concatenate empty lists of local vectors.")

        # Check that all vectors have compatible shapes
        expected_dofs = local_vectors_list[0].shape[1]
        for i, vector in enumerate(local_vectors_list):
            if vector.shape[1] != expected_dofs:
                raise ValueError(
                    f"All local vectors must have the same number of degrees of "
                    f"freedom. Expected shape (*, {expected_dofs}), "
                    f"but vector {i} has shape {vector.shape}."
                )
        for i, indices in enumerate(local_indices_list):
            if indices.shape[1] != expected_dofs:
                raise ValueError(
                    f"All local indices must have the same number of degrees of "
                    f"freedom. Expected shape (*, {expected_dofs}), "
                    f"but indices {i} has shape {indices.shape}."
                )

        offsets: list[int] = [0]
        for local_indices in local_indices_list:
            offsets.append(offsets[-1] + local_indices.shape[0])

        all_local_vectors = dr.zeros(
            dtype=mi.TensorXf, shape=(offsets[-1], local_vectors_list[0].shape[1])
        )
        all_local_indices = dr.zeros(
            dtype=mi.TensorXu, shape=(offsets[-1], local_indices_list[0].shape[1])
        )
        mappings = [mapping for _, mapping in composite_iga_spline.patches_with_mapping]

        for i, (local_indices, local_vectors, mapping) in enumerate(
            zip(local_indices_list, local_vectors_list, mappings)
        ):
            start = offsets[i]
            end = offsets[i + 1]

            all_local_vectors[start:end, :] = dr.ravel(local_vectors)
            all_local_indices[start:end, :] = dr.ravel(
                mi.TensorXu(mapping[local_indices])
            )

        return all_local_indices, all_local_vectors

    @staticmethod
    def _transform_basis_derivative(
        basis_derivative: "mi.TensorXf",
        inv_metric: "mi.TensorXf",
    ) -> "mi.TensorXf":
        """
        Performs matrix-matrix multiplication to transform basis function derivatives
        from parametric coordinates to almost physical coordinates on the manifold.

        This method applies the chain rule to transform derivatives of basis functions
        from the parametric domain (u, v) to the physical domain (x, y) on the manifold.
        It does not complete this task as the full transformation requires an additional
        multiplication with the Jacobian of the mapping. This is avoided as it is not
        required for stiffness matrix assembly in the current context. Thus, it applies
        only a transformation G^{-1} grad_param phi, where grad_param phi is given
        by `basis_derivative`.

        :returns: Transformed basis function derivatives with shape
            (n_knot_spans, n_dofs, 2)

        :raises ValueError: If the shape of `inv_metric` does not match the expected
            shape based on the number of knot spans in `basis_derivative`.
        """
        if inv_metric.shape != (basis_derivative.shape[0], 2, 2):
            raise ValueError(
                f"inv_metric must have shape ({basis_derivative.shape[0]}, 2, 2), "
                f"got {inv_metric.shape}."
            )

        transformed_basis_function_derivative = dr.empty(
            dtype=mi.TensorXf, shape=basis_derivative.shape
        )
        transformed_basis_function_derivative[:, :, 0] = dr.ravel(
            inv_metric[:, 0, 0][:, None] * basis_derivative[:, :, 0]
            + inv_metric[:, 0, 1][:, None] * basis_derivative[:, :, 1]
        )
        transformed_basis_function_derivative[:, :, 1] = dr.ravel(
            inv_metric[:, 1, 0][:, None] * basis_derivative[:, :, 0]
            + inv_metric[:, 1, 1][:, None] * basis_derivative[:, :, 1]
        )

        return transformed_basis_function_derivative

    @staticmethod
    def _local_matrix_scatter_indices(
        num_knot_spans: int, num_dofs: int, offset: int
    ) -> "mi.TensorXu":
        """
        Generates scatter indices for assembling local matrices using vectorized
        operations.

        This method creates index arrays that allow efficient assembly of local
        matrices using DrJit's scatter operations. The indices map from a linearized
        representation of local matrix entries to the appropriate positions in the
        flattened local matrix tensor.

        The scatter indices are designed to work with the local matrix assembly
        process where contributions from different quadrature points are accumulated
        into local matrices. The `offset` parameter allows for row-wise assembly of
        the matrix.

        :returns: Scatter indices with shape (`num_knot_spans` * `num_dofs`,) that map
            local matrix entries to their correct positions in the flattened local
            matrix tensor.

        :raises ValueError: If parameters are non-positive or if offset is out of the
            valid range.
        """
        if num_knot_spans <= 0:
            raise ValueError(f"num_knot_spans must be positive, got {num_knot_spans}.")

        if num_dofs <= 0:
            raise ValueError(f"num_dofs must be positive, got {num_dofs}.")

        if offset < 0 or offset >= num_dofs:
            raise ValueError(f"offset must be in range [0, {num_dofs}), got {offset}.")

        base_indices = (
            num_dofs**2 * dr.arange(dtype=mi.UInt32, size=num_knot_spans)
            + offset * num_dofs
        )

        repeated_base = dr.repeat(value=base_indices, count=num_dofs)
        tiled_offsets = dr.tile(
            dr.arange(dtype=mi.UInt32, size=num_dofs), count=num_knot_spans
        )

        return mi.TensorXu(repeated_base + tiled_offsets)

    @staticmethod
    def _scatter_values_to_local_matrices(
        local_matrices: "mi.TensorXf",
        values: "mi.TensorXf",
        num_knot_spans: int,
        num_dofs: int,
        row: int,
    ) -> None:
        """
        Scatters values into the specified `row` of all local matrices (all knot spans).

        This method uses DrJit's scatter_add operation to efficiently accumulate values
        into the correct positions of the local matrix tensor for each knot span and
        degree of freedom. The scatter indices are computed such that the provided
        values are added to the appropriate row entries of the local matrices. Note
        that there is never collision in the indices, as each row corresponds to a
        unique knot span.

        This is typically used during the assembly of local element matrices in
        isogeometric analysis, where contributions from different quadrature points
        or basis functions are accumulated into the local stiffness or mass matrices.

        :raises ValueError: If the input shapes are incompatible or if the row index
            is out of bounds.
        """
        scatter_indices = IGAManifoldProblem._local_matrix_scatter_indices(
            num_knot_spans=num_knot_spans, num_dofs=num_dofs, offset=row
        )
        dr.scatter_add(
            target=local_matrices.array,
            value=dr.ravel(values),
            index=dr.ravel(scatter_indices),
            mode=dr.ReduceMode.NoConflicts,
        )
