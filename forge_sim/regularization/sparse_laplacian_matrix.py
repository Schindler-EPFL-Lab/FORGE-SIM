import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt


class SparseLaplacianMatrix:
    """
    A class that represents a sparse Laplacian matrix of a graph.

    It primarily provides functionality to compute L x / d, where L is the
    combinatorial Laplacian matrix constructed from faces, x is a tensor of
    positions, normals, or something else that can be averaged over faces, and d is the
    sum of distances between the nodes in the mesh.The Laplacian
    applied to each component separately.

    The sum of distances is not divided by the number of neighbors on purpose
    because the combinatorial Laplacian is not normalized (n_neigh * my_val - sum(
    neigh_val should be divided by n_neigh but isn't by  definition of L).
    Normalization is important to avoid mesh dependent weighting.

    The goal is to reuse Laplacian computation and the averaging logic that effectively
    computes a Laplacian vector product that is then squared and summed.

    The combinatorial (topological) Laplacian of an undirected graph is defined as
    L = D - A,
    where D is the degree of the nodes (a diagonal matrix) and A is the adjacency
    matrix. It is the simplest form of Laplacian that can be computed purely from the
    connectivity of the mesh. This is beneficial as it does not require recomputing it
    after every vertex update.

    A more expensive, but more accurate version would be the cotangent Laplacian,
    which takes edge lengths and angles into account. It seems that up to now most
    literature did not use it, and the main reason seems to be the isotropic nature of
    the meshes that are being optimized, reducing the benefit of the cotangent
    Laplacian. It might be worth looking into it in the future.
    """

    ZERO_LENGTH_TOLERANCE = 1e-6
    """Defines the minimum distance between nodes to avoid division by zero."""

    def __init__(self, faces: npt.NDArray[np.uint32]) -> None:
        """
        Builds the sparse combinatorial Laplacian matrix from the given faces.

        The faces are a 2D array of shape (F, C), where F is the number of faces
        and C is the number of nodes per face (e.g., 3 for triangles, 4 for quads).
        It must have at least 2 vertices per face.

        :raises ValueError: If the faces are not a 2D array or have less than 2 vertices
            per face or if the weight is negative.
        """
        if len(faces.shape) != 2 or faces.shape[1] < 2:
            raise ValueError(
                "Faces must be a 2D array with at least 2 vertices per face."
            )
        # Compute the Laplacian matrix indices and values
        indices, values = self.combinatorial_laplacian(faces)

        # Convert to Mitsuba tensors for DrJIT compatibility
        self._row_indices = dr.ravel(mi.TensorXu(indices[0]))
        self._col_indices = dr.ravel(mi.TensorXu(indices[1]))
        self._laplacian_values = dr.reshape(
            dtype=mi.TensorXf, value=mi.TensorXf(values), shape=(-1, 1)
        )

    def apply_to(
        self, values_to_average: "mi.TensorXf", positions: "mi.TensorXf"
    ) -> "mi.TensorXf":
        """
        Computes the Laplacian loss || L x / d ||^2 for the values to average.

        The algorithm computes the Laplacian vector product L x, where L is the
        Laplacian matrix and x are the values to average (e.g., positions, normals,
        etc.). The algorithm works as follows:

            1. Computes the difference between the positions of all edges, takes the
               Euclidean distances and gathers the sum for each node index.
            2. expands x, shaped (N, C), into (E, C) where E is the number of edges
               in the graph.
            3. Multiplied by the weight of the Laplacian matrix at that index.
            4. scatter add (reduce) the values to the corresponding row indices

        :raises ValueError: If the nodes to average do not have shape (N, C) where N is
            the number of nodes and C is the number of components per node.
        """
        if len(values_to_average.shape) != 2:
            raise ValueError(
                f"Expected `values_to_average to have shape (N, C), "
                f"got {values_to_average.shape}."
            )
        if len(positions.shape) != 2:
            raise ValueError(
                f"Expected `positions` to have shape (N, D), got {positions.shape}."
            )
        if values_to_average.shape[0] != positions.shape[0]:
            raise ValueError(
                "Number of elements in `values_to_average` and `positions` must match."
            )

        num_values = values_to_average.shape[0]

        # 1. Compute the difference vector and distances.
        diff_vector = positions[self._row_indices, :] - positions[self._col_indices, :]
        # Mind the safe_sqrt - the GPU code does not work otherwise!
        # Use it whenever it is possible to get sqrt(0). Here, this happens for
        # diagonal elements (row_index = col_index) and destroys the gradients
        distances = dr.safe_sqrt(dr.sum(diff_vector * diff_vector, axis=1))

        distance_sum = dr.zeros(dtype=mi.TensorXf, shape=num_values)
        dr.scatter_add(
            target=distance_sum.array,
            value=dr.ravel(distances),
            index=self._row_indices,
        )
        # Avoid division by zero by setting small distances to a minimum value
        distance_sum = dr.maximum(distance_sum, self.ZERO_LENGTH_TOLERANCE)

        # 2. and 3. expand the values to average and multiply by the Laplacian weights.
        weighted_values = (
            self._laplacian_values * values_to_average[self._col_indices, :]
        )
        node_laplacian = dr.zeros(dtype=mi.TensorXf, shape=values_to_average.shape)

        # 4. Scatter add the weighted nodes to the corresponding row indices.
        for coord in range(values_to_average.shape[1]):
            buffer = dr.zeros(dtype=mi.TensorXf, shape=num_values)
            dr.scatter_add(
                target=buffer.array,
                value=dr.ravel(weighted_values[:, coord]),
                index=self._row_indices,
            )
            node_laplacian[:, coord] = dr.ravel(buffer / distance_sum)

        return node_laplacian

    @staticmethod
    def combinatorial_laplacian(
        faces: npt.NDArray[np.uint32],
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """
        Computes the combinatorial (topological) Laplacian based on the face
        connectivity. Handles undirected graphs, triangle meshes, quad meshes, etc.

        The combinatorial (topological) Laplacian of an undirected graph is defined as
        L = D - A,
        where D is the degree of the nodes (a diagonal matrix) and A is the adjacency
        matrix. It is the simplest form of Laplacian that can be computed purely from
        the connectivity of the mesh.

        It returns two index arrays that can be used to construct a sparse COO matrix.
        Note that the diagonal entries are repeated degree times for each vertex.

        :returns: Tuple (`indices`, `values`) where:

            - `indices` is a 2D array of shape (2, E) where the first row contains
              the row indices and the second row contains the column indices of the
              non-zero entries of the Laplacian matrix.
            - `values` is a 1D array of shape (E,) containing -1 for off-diagonal
              entries and 1 for diagonal entries of the Laplacian matrix.

            The number of indices is proportional to the number of edges E in the graph.

        :raises ValueError: If the input `faces` is not a 2D array or has less than
            2 nodes per face.
        """
        if len(faces.shape) != 2 or faces.shape[1] < 2:
            raise ValueError("Faces must be a 2D array with at least 2 nodes per face.")

        # Constructs edges. Example effect for triangles:
        # ii = faces[:, [1, 2, 0]].flatten(), jj = faces[:, [2, 0, 1]].flatten()
        # ii are the row indices, jj are the column indices of the adjacency matrix.
        ii = faces[:, np.arange(faces.shape[1])].flatten()
        jj = faces[:, np.roll(np.arange(faces.shape[1]), -1)].flatten()

        adj_indices = np.unique(
            np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0),
            axis=1,
        )
        adj_values = np.ones(adj_indices.shape[1], dtype=np.float32)

        # Diagonal indices, duplicated as many times as the connectivity of each index
        diag_indices = np.stack((adj_indices[0], adj_indices[0]), axis=0)

        # Build the sparse matrix
        idx = np.concatenate((adj_indices, diag_indices), axis=1)
        values = np.concatenate((-adj_values, adj_values))

        return np.array(idx, dtype=np.uint32), np.array(values, dtype=np.float32)
