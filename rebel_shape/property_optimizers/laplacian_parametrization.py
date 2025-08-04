import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
from mitsuba.python.ad.largesteps import SolveCholesky

from rebel_shape.regularization.sparse_laplacian_matrix import SparseLaplacianMatrix


class LaplacianParametrization:
    """
    Implementation of the algorithm described in the paper "Large Steps in
    Inverse Rendering of Geometry" (Nicolet et al. 2021).

    It consists of computing a latent variable u = (I + `smoothing_strength` L) x
    from Cartesian node positions x. L is the (combinatorial) Laplacian matrix of
    the input graph defined by nodes and faces. The combinatorial Laplacian is chosen
    because it only depends on the topology (connectivity) and not on node positions,
    obviating the need to recompute it after each node position update.

    Optimizing the parametrized variables instead of the node positions directly
    diffuses gradients, reducing their magnitude and spreading them out over the
    surface that is optimized. This is especially useful for silhouette gradients which
    are often very large and sparse, leading to unstable optimization.

    It maintains a Cholesky factorization of matrix (I + `smoothing_strength` L) that
    defines the parametrization, allowing for fast solves of the linear system.
    Thus, it can efficiently convert between the Cartesian and parametrized
    representations. Both transformations are differentiable, allowing them to be
    used in a gradient-based optimization.

    Note that selecting the right value for `smoothing_strength` is crucial for
    good optimization results. If the value is too low, the gradients are too strong,
    while if it is too high, the optimized shape is too smooth. The factor
    depends on the total number of nodes in the mesh - larger node counts require
    larger values to achieve the same smoothing effect.

    This class was copied from "mitsuba.ad.LargeSteps" and adapted to our needs.
    Specifically, there was a need to optimize quads instead of triangles, and the only
    way to achieve this was by copying the code and modifying the Laplacian
    construction to work with both quads and triangles.

    For more information on the exact mechanism and how it works, look up the paper by
    Nicolet et al. "Large Steps in Inverse Rendering of Geometry" (2021).
    """

    def __init__(
        self, nodes: "mi.TensorXf", faces: "mi.TensorXu", smoothing_strength: float
    ) -> None:
        """
        Initializes the Laplacian parametrization with a graph defined by `nodes`,
        `faces` and `smoothing_strength`.

        `nodes` are the positions of nodes in the graph (e.g. vertices of a mesh) of
        shape (N, 3) where N is the number of nodes. `faces` are the indices of the
        nodes that form the connectivity, with shape (F, C) where F is the number
        of faces and C is the number of nodes per face (e.g. 3 for triangles, 4 for
        quads).

        `smoothing_strength` is a non-negative float that controls how strongly the
        gradients are smoothed. A value of 0 means no smoothing, while larger values
        will result in stronger smoothing. An appropriate value depends on the
        number of nodes and has to be determined empirically.

        :raises ValueError: If `nodes` is not a 2D array with shape (N, 3), if `faces`
            is not a 2D array with shape (F, C) where C >= 2, or if `smoothing_strength`
            is a negative number.
        """
        if len(nodes.shape) != 2 or nodes.shape[1] != 3:
            raise ValueError("`nodes` must be a 2D array with shape (N, 3)!")
        if len(faces.shape) != 2 or faces.shape[1] < 2:
            raise ValueError(
                "`faces` must be a 2D array with shape (F, C) where C >= 2!"
            )
        if smoothing_strength < 0:
            raise ValueError("`smoothing_strength` must be a non-negative number!")

        # Remove duplicates due to e.g. UV seams or face normals.
        # This is necessary to avoid seams opening up during optimisation
        unique_nodes, node_index, node_inverse = np.unique(
            nodes.numpy(), return_index=True, return_inverse=True, axis=0
        )
        node_inverse = node_inverse.flatten()
        unique_faces = node_inverse[faces.numpy()]

        self._node_index = dr.ravel(mi.TensorXu(node_index))
        """Mapping from the original node indices to the unique node indices."""
        self._node_inverse_mapping = dr.ravel(mi.TensorXu(node_inverse))
        """Mapping from the unique node indices back to the original node indices."""
        self._num_nodes = unique_nodes.shape[0]
        """
        Number of unique nodes in the graph, i.e. the number of rows in the 
        Laplacian matrix.
        """

        self._smoothing_strength = smoothing_strength
        """The strength of the smoothing applied to the gradients."""
        self._unique_faces = mi.TensorXu(unique_faces)
        """Unique faces of the graph."""

        # Initialize the solver and the Laplacian matrix
        self._init_laplacian_and_solver()

    def _init_laplacian_and_solver(self) -> None:
        """
        Initializes the Cholesky solver with the Large Steps Laplacian.

        Internally, it calls the `large_steps_laplacian` method to compute the
        combinatorial Laplacian matrix, and then constructs the Cholesky solver
        using the indices and values of the matrix.

        Exceptionally, this method contains import statements for cholespy.
        This is necessary because `CholeskySolver` is bound to the Mitsuba variant.

        :raises ImportError: If the `cholespy` package is not installed.
        """
        # Due to Mitsuba magic, these cannot be imported at the top of the file.
        # They are bound to the Mitsuba variant, so they need to be imported here.
        from cholespy import CholeskySolverF as CholeskySolver
        from cholespy import MatrixType

        indices, values = self.large_steps_laplacian(
            num_nodes=self._num_nodes,
            faces=self._unique_faces.numpy(),
            smoothing_strength=self._smoothing_strength,
        )
        indices_unique, inverse_idx = np.unique(indices, axis=1, return_inverse=True)
        inverse_idx = inverse_idx.flatten()

        # For compatibility with CholeskySolver, indices must be signed integers
        self._ls_rows = mi.TensorXi(indices_unique[0])
        self._ls_cols = mi.TensorXi(indices_unique[1])

        # CholespySolver requires double precision for the data array
        ls_values = dr.zeros(dtype=mi.TensorXd, shape=(indices_unique.shape[1],))

        dr.scatter_reduce(
            op=dr.ReduceOp.Add,
            target=ls_values.array,
            value=dr.ravel(mi.TensorXd(values)),
            index=dr.ravel(mi.TensorXu(inverse_idx)),
        )

        self._solver = CholeskySolver(
            self._num_nodes, self._ls_rows, self._ls_cols, ls_values, MatrixType.COO
        )
        self._ls_values = dr.reshape(
            dtype=mi.TensorXf, value=mi.TensorXf(ls_values), shape=(-1, 1)
        )

    @staticmethod
    def large_steps_laplacian(
        num_nodes: int, faces: npt.NDArray[np.uint32], smoothing_strength: float
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """
        Computes the combinatorial Laplacian and returns the system matrix
        (I + `smoothing_strength` L) for the Large Steps Laplacian parametrization.

        It uses the `SparseLaplacianMatrix.combinatorial_laplacian` method to compute
        the "standard" combinatorial Laplacian matrix L, and inserts extra entries
        to create the full matrix.

        :returns: The indices and values of matrix (I + `smoothing_strength` L).
        :raises ValueError: If `faces` is not a 2D array with shape (F, C) where C >= 2,
            or if `num_nodes` is not a positive integer.
        """
        if faces.ndim != 2 or faces.shape[1] < 2:
            raise ValueError(
                "`faces` must be a 2D array with shape (F, C) where C >= 2!"
            )
        if num_nodes < 1:
            raise ValueError("`n_nodes` must be a positive integer!")
        indices, values = SparseLaplacianMatrix.combinatorial_laplacian(faces)

        # Create a sparse identity matrix diagonal elements
        diag = np.stack((np.arange(num_nodes), np.arange(num_nodes)), axis=0)
        diag_values = np.ones(num_nodes, dtype=np.float32)

        # Combine Laplacian and identity components: I + `smoothing_strength` * L
        indices = np.concatenate((indices, diag), axis=1)
        values = np.concatenate((values * smoothing_strength, diag_values), axis=0)

        return indices, values

    def to_parametric(self, nodes_flat: "mi.Float") -> "mi.Float":
        """
        Convert flattened node Cartesian coordinates `nodes_flat` to their parametric
        form: u = (I + `smoothing_strength` L) x.

        The conversion is cheap as it consists of a matrix-vector multiplication with
        a sparse matrix which is O(num_edges) in complexity,

        This method typically only needs to be called once, at the beginning of the
        optimization, to obtain a latent (parametric) representation of the variable
        that is optimized, like the vertex positions or control points.

        :returns: A flat array of Laplacian parametrized values of shape (N * 3,).
        """
        # Perform a manual matrix-vector multiplication with the Large Steps Laplacian
        nodes = dr.reshape(
            dtype=mi.TensorXf,
            value=nodes_flat,
            shape=(-1, 3),
        )

        unique_nodes = dr.empty(dtype=mi.TensorXf, shape=(self._num_nodes, 3))
        for comp in range(3):
            unique_nodes[:, comp] = dr.ravel(nodes[self._node_index, comp])

        # Expand the points to the shape of the flat matrix representation
        row_product = unique_nodes[dr.ravel(self._ls_rows), :] * self._ls_values

        # Finish the matrix-vector multiplication by summing the contributions
        parametric = dr.empty(dtype=mi.TensorXf, shape=(self._num_nodes, 3))

        for comp in range(3):
            buffer = dr.zeros(dtype=mi.TensorXf, shape=(self._num_nodes,))
            dr.scatter_add(
                target=buffer.array,
                value=dr.ravel(row_product[:, comp]),
                index=dr.ravel(self._ls_cols),
            )
            parametric[:, comp] = dr.ravel(buffer)

        return dr.ravel(parametric)

    def from_parametric(self, parametric_flat: "mi.Float") -> "mi.Float":
        """
        Convert the `parametric_flat` values back to Cartesian coordinates:
        x = (I + `smoothing_strength` L)^{-1} u.

        This is done by solving the linear system (I + λL) v = u using the
        previously computed Cholesky factorization. Since the Laplacian does not
        depend on the node positions, the factorization does not change, allowing
        for a very fast solve.

        This method is typically called at each iteration of the optimization,
        to update the vertex coordinates or control points of the mesh.

        :returns: A flat array of Cartesian coordinates of shape (N * 3,).
        """
        unique_nodes = dr.unravel(
            dtype=mi.Point3f,
            array=dr.custom(
                SolveCholesky,
                self._solver,
                mi.TensorXf(parametric_flat, (self._num_nodes, 3)),
            ).array,
        )

        return dr.ravel(
            dr.gather(
                dtype=mi.Point3f, source=unique_nodes, index=self._node_inverse_mapping
            )
        )
