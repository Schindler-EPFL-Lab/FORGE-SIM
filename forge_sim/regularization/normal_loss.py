import drjit as dr
import mitsuba as mi

from forge_sim.meshing.mesh import Mesh
from forge_sim.regularization.regularization import Regularization
from forge_sim.regularization.sparse_laplacian_matrix import SparseLaplacianMatrix


class NormalLoss(Regularization):
    """
    A regularization loss that encourages the vertex normals of a mesh to be
    similar to the average of their neighboring normals.

    However, there is a form of "malicious compliance" where the vertices bunch
    together, making it appear as the neighboring normals are similar, but the
    surface is not smooth. This can lead to big creases in the surface that cannot be
    removed by this loss alone. It is best used in conjunction with `PositionLoss`.
    The weights of these two losses should be balanced to avoid either one from
    dominating excessively.
    """

    def __init__(self, mesh: Mesh, weight: float = 1.0) -> None:
        """
        Returns an instance of normal loss for the provided `mesh`.

        Mesh faces are used to construct the combinatorial Laplacian matrix, and the
        vertex normals are used as the values to average.

        Should not be used alone, but in conjunction with `PositionLoss` to ensure
        that the surface does not develop creases due to the vertices bunching together.

        :raises ValueError: If the weight is negative.
        """
        # Check for negative weight done in the `Regularization` base class
        super().__init__(weight=weight)
        self._mesh = mesh
        self._laplacian = SparseLaplacianMatrix(faces=self._mesh.faces.numpy())

    def evaluate(self) -> "mi.TensorXf":
        laplacian_product = self._laplacian.apply_to(
            values_to_average=self._mesh.vertex_normals,
            positions=self._mesh.vertices,
        )
        dot_product = dr.sum(laplacian_product * laplacian_product, axis=1)

        return self.weight * dr.mean(dot_product)
