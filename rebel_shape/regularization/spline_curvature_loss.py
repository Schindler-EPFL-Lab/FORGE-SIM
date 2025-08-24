import drjit as dr
import mitsuba as mi

from rebel_shape.isogeometric_analysis.iga_manifold_problem import IGAManifoldProblem
from rebel_shape.meshing.tessellated_spline_surface import TessellatedSplineSurface
from rebel_shape.regularization.regularization import Regularization


class SplineCurvatureLoss(Regularization):
    r"""
    Curvature loss based on minimizing the Willmore energy.

    According to Wikipedia: "Willmore energy shows how much a given surface deviates
    from a round sphere." This loss reduces sharp peaks and other artifacts that can
    break the simulation.

    The value of this loss is given by the surface integral
    ```math
    W = \int_S (H^2 - K) dA,
    ```
    where H is the mean curvature and K is the Gaussian curvature.

    Let $k_1$ and $k_2$ be the principal curvatures. One can rewrite:

    ```math
    W = \int_S \left(\frac{1}{4} (k_1 + k_2)^2 - k_1 k_2\right) dA
    = \int_S \frac{1}{4} (k_1 - k_2)^2 dA
    ```

    Intuitively, this regularization severely discourages the formation of ridges which
    exhibit significantly different values of curvature along the two parametric
    directions.
    """

    def __init__(
        self, tessellated_spline: TessellatedSplineSurface, weight: float = 1.0
    ) -> None:
        """
        Initializes the `SplineCurvatureLoss` with a tessellated spline surface
        `tessellated_spline` and a `weight`, which is divided by the number of patches
        in the spline surface.
        """
        super().__init__(weight=weight / len(tessellated_spline))
        self._tessellated_spline = tessellated_spline

    def evaluate(self) -> "mi.TensorXf":
        r"""
        Evaluate the spline curvature loss by averaging the individual patch curvature
        losses.

        Gauss-Legendre quadrature is applied to calculate the integral. According to
        Wikipedia, it approximates the definite integral of a function f by a finite
        weighted sum:

        ```math
        \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)
        ```

        where
        - $n$ is the number of sample points used,
        - $w_i$ are the quadrature weights, and
        - $x_i$ are the roots of the nth Legendre polynomial.

        A good set of slides explaining Gauss-Legendre quadrature can be found here:
        https://homepages.dias.ie/leitner/MS213-material-for-website/NA9_F_Gauss_Quad.pdf

        The approximated expression is:
        ```math
            E = \sum_i \sum_j E_{i,j}
        ```

        where:

        - $i, j$ are indices of the knot spans in the u and v directions.
        - $E_{i,j}$ is the Willmore energy of a small piece of the surface, defined as a
        projection from a rectangle ($\text{rectangle}_{i,j}$) in parametric coordinates
        to the surface.

        The expression for E_{i,j} is:
        ```math
            E_{i,j} = \frac{1}{4} (\xi_{i+1}^* - \xi_{i}^*) (\eta_{j+1}^* - \eta_{j}^*)
                    \sum_{k=0}^{n_k} \sum_{l=0}^{n_l} w_{k,n_k} w_{l,n_l}
                    \cdot \frac{1}{4} (k_1(\eta_k, \xi_l) - k_2(\xi_k, \eta_l))^2 \cdot
                    g(\xi_k, \eta_l)
        ```
        where:

        - $(\xi_{i}^*, \xi_{i+1}^*)$ and $(\eta_{j}^*, \eta_{j+1}^*)$ are intervals in
        the parametric coordinates that define $\text{rectangle}_{i,j}$.
        - $eta_k, xi_l$ are evaluation points of the Gauss-Legendre quadrature on
        $\text{rectangle}_{i,j}$. There are n_k points in the u direction and n_l points
        in the v direction.
        - $k_1, k_2$ are the principal curvatures.
        - $w_{k,n_k}, w_{l,n_l}$ are Gauss-Legendre quadrature weights.
        - $g(\xi_k, \eta_l) = \sqrt{\det{(\text{metric})}}$, which appears due to the
        integration rules for parametric surfaces (see: https://en.wikipedia.org/wiki/Surface_integral).

        There is an additional term:
        ```math
            \frac{1}{4} (\xi_{i+1}^* - \xi_{i}^*) (\eta_{j+1}^* - \eta_{j}^*)
        ```

        which appears because the integration interval has been changed from
        $(-1, 1) \times (-1, 1)$ to $(\xi_{i}^*, \xi_{i + 1}^*) \times (\eta_{j}^*,
        \eta_{j + 1}^*).


        Intergration order was heuristically chosen as (2 * spline degree). Typically
        (spline degree + 1) is enough, but in this case a higher order should be chosen
        to stabilize the outcome values.

        :returns Tensor of shape (1,) containing the regularization loss value.
        """
        total_loss = mi.TensorXf(0.0)

        for patch in self._tessellated_spline.composite_spline:
            knot_span_handler = IGAManifoldProblem.KnotSpanHandler(
                patch=patch, integration_order=(2 * patch.degree)
            )

            param_coords, weight = knot_span_handler.full_gl_coords_and_weights

            jacobian = patch.evaluate_jacobian(param_coords)
            normal = patch.evaluate_normals(param_coords)
            hessian = patch.evaluate_hessian(param_coords)

            dr.schedule(normal, jacobian, hessian)

            metric = patch.metric_tensor(jacobian)
            inv_metric, det_metric = patch.inv_metric_tensor(metric)
            second_form = patch.second_fundamental_form(hessian=hessian, normal=normal)

            k1, k2 = patch.principal_curvatures(
                inv_metric_tensor=inv_metric, second_form=second_form
            )

            total_loss += 0.25 * dr.sum(
                (k1 - k2) ** 2 * weight * dr.safe_sqrt(det_metric)
            )

        return total_loss * self.weight
