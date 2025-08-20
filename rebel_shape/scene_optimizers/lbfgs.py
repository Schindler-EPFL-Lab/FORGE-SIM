from collections.abc import Callable

import drjit as dr
import mitsuba as mi

from rebel_shape import logger
from rebel_shape.scene_optimizers.scene_optimizer import MitsubaOptimizationType


class LBFGS(mi.ad.Optimizer):
    """
    Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer implementation
    that extends the Mitsuba base optimizer class.

    L-BFGS is a quasi-Newton optimization method that approximates the inverse Hessian
    matrix using a limited history of previous gradient and parameter updates. This
    implementation supports both Armijo and Wolfe line search conditions for step
    size determination.

    The optimizer maintains separate optimization state for each parameter key,
    including history buffers for state differences, gradient differences, and
    their dot products. When the history exceeds the specified size, older entries
    are automatically discarded to maintain memory efficiency.
    Negative curvature updates are skipped.

    Line search is performed using backtracking with configurable iteration limits.
    The algorithm can use either the Armijo condition alone or both Armijo and Wolfe
    conditions for more robust convergence.

    This algorithm is typically used for smooth, continuous optimization problems,
    but it was found useful for shape optimization tasks with splines due to the
    robustness of the line search procedure and the ability to preserve shape smoothness
    during optimization.

    However, it is much more computationally expensive than Adam and other first-order
    methods, especially if the step size is too large and the line search has to be
    performed many times - images have to be re-rendered for each line search
    iteration, which can be very slow. Using the Wolfe curvature condition is even
    slower as the gradient has to be computed for each line search iteration.

    The implementation is based on the code from Dr.TVAM
    https://github.com/rgl-epfl/drtvam/blob/f1348f58f194e83dcca0c47ef17d3a74d402f047/src/drtvam/lbfgs.py
    and is written by the same team that developed Dr.Jit and Mitsuba 3.
    """

    CURVATURE_EPSILON: float = 1e-8

    def __init__(
        self,
        learning_rate: float = 1.0,
        history_size: int = 10,
        params: dict[str, MitsubaOptimizationType] | None = None,
        use_wolfe_curvature: bool = False,
        line_search_iterations: int = 20,
    ) -> None:
        """
        Initialize the L-BFGS optimizer with the specified parameters.

        `learning_rate` rate is the default step size for all variables; it is the
        initial step length that is adjusted during line search.

        `history_size` is the number of previous states that are used to compute a
        quasi-Newton approximation of the inverse Hessian matrix that in turn is used
        to compute the new search direction. Values older than this size are discarded
        from the history to save memory. Typical values are between 5 and 100,
        depending on the problem complexity and memory constraints.

        `params` is a dictionary of initial parameters to optimize. If not provided,
        the optimizer will start with an empty set of parameters and they have to be
        set later by simply assigning a new key to the optimizer (Mitsuba `ad.Optimizer`
        supports dictionary-like access to parameters).

        `use_wolfe_curvature` determines whether to use the Wolfe condition on the
        gradient when conducting line search or to just use the Armijo condition.
        In most cases it is not recommended to use Wolfe condition as it is both more
        expensive to evaluate and can lead to stalling of the optimization process.

        `line_search_iterations` is the number of iterations of backtracking line search
        to run. It determines how many times the step size is halved until the Armijo
        condition (and Wolfe condition if enabled) is satisfied. Values higher than 20
        are typically not needed - taking a very small step (`learning_rate` / 2^20)
        in a wrong direction is unlikely to destabilize the optimization process.

        Note that it is necessary to set the `line_search_fn` property before
        starting the optimization process.
        """
        if learning_rate <= 0.0:
            raise ValueError("`learning_rate` has to be a positive real number")

        if history_size < 1:
            raise ValueError("`history_size` has to be a positive integer")
        if history_size < 5:
            logger.warning(
                "`history_size` is set to a very low value, which may lead to "
                + "poor convergence behavior. Consider increasing it."
            )
        if history_size > 100:
            logger.warning(
                "`history_size` is set to a very high value, which may lead to "
                + "higher memory usage and slower optimization. Consider reducing it."
            )

        if line_search_iterations < 1:
            raise ValueError("`line_search_iterations` has to be a positive integer")
        if line_search_iterations < 5:
            logger.warning(
                "`line_search_iterations` is set to a low value, which may lead to "
                + "poor convergence behavior. Consider increasing it."
            )
        if line_search_iterations > 30:
            logger.warning(
                "`line_search_iterations` is set to a very high value, which may lead "
                + "to increased computation time. Consider reducing it."
            )

        super().__init__(lr=learning_rate, params=params)

        self._use_wolfe_curvature = use_wolfe_curvature
        """
        Whether to use the Wolfe condition on the gradient when conducing line search 
        or to just use the Armijo condition. Typically not recommended as it 
        can lead to stalling of the optimization process.
        """
        self._history_size = history_size
        """Number of previous states to keep in history for update."""

        self._previous_params: dict[str, MitsubaOptimizationType] = {}
        """Previous state for each optimization variable."""
        self._previous_gradient: dict[str, MitsubaOptimizationType] = {}
        """Previous gradients for each optimization variable."""
        self._state_difference: dict[str, list[MitsubaOptimizationType]] = {}
        """Difference between current and previous state for each variable."""
        self._gradient_differences: dict[str, list[MitsubaOptimizationType]] = {}
        """Difference between current and previous gradients for each variable."""
        self._iterations: dict[str, int] = {}
        """Number of iterations for each variable."""
        self._gradient_diff_dot_state_diff: dict[
            str, list[MitsubaOptimizationType]
        ] = {}
        """
        Dot product of current and previous gradient differences and state differences.
        """

        self._line_search_fn: (
            Callable[[dict[str, MitsubaOptimizationType]], MitsubaOptimizationType]
            | None
        ) = None
        """
        Function used for line search, should take a dictionary of variables as input
        and return the loss and gradients for each variable.
        """

        self._line_search_iterations = line_search_iterations
        """Number of iterations of backtracking line search to run."""

    def reset(self, variable_key: str) -> None:
        """
        Reset the optimization history for a specific variable with key `variable_key`.

        Clears all stored state differences, gradient differences, and dot products
        for the specified variable. This is automatically called when a variable
        is first encountered or when its shape changes.
        """
        self._state_difference[variable_key] = []
        self._gradient_differences[variable_key] = []
        self._gradient_diff_dot_state_diff[variable_key] = []
        self._iterations[variable_key] = 0

    def _update_history(self, variable_key: str) -> None:
        """
        Update the L-BFGS history with current parameter and gradient information.

        Computes differences between current and previous states/gradients and
        stores them in the history buffers for a variable with key `variable_key`.

        If the computed curvature is negative, it skips the update.

        Automatically manages history size by removing the oldest entries when the
        buffer exceeds the specified limit. It also schedules DrJit operations for
        the computed differences and updates the iteration counter for the variable.

        This method is called automatically during the optimization step and typically
        should not be called manually.
        """
        params = dr.ravel(dr.detach(self.variables[variable_key]))
        params_gradient = dr.ravel(self.variables[variable_key].grad)
        dr.schedule(params, params_gradient)

        # Update history if not the first iteration
        if self._iterations[variable_key] > 0:
            s = params - self._previous_params[variable_key]
            y = params_gradient - self._previous_gradient[variable_key]
            dr.schedule(s, y)

            ys = dr.dot(y, s)

            # Only apply updates if the curvature is positive
            if ys > self.CURVATURE_EPSILON:
                # Discard entries that are too old
                if self._iterations[variable_key] > self._history_size:
                    self._state_difference[variable_key].pop(0)
                    self._gradient_differences[variable_key].pop(0)
                    self._gradient_diff_dot_state_diff[variable_key].pop(0)

                self._state_difference[variable_key].append(s)
                self._gradient_differences[variable_key].append(y)
                self._gradient_diff_dot_state_diff[variable_key].append(ys)

                self._iterations[variable_key] += 1
            else:
                logger.debug(
                    f"Rejected update for {variable_key} due to negative "
                    + f"curvature {ys} <= {self.CURVATURE_EPSILON}. "
                )
        else:
            # First iteration, initialize history
            self._iterations[variable_key] = 1

        # Update previous state and gradient
        self._previous_params[variable_key] = params
        self._previous_gradient[variable_key] = params_gradient

    def _apply_inverse_hessian(
        self, variable_key: str, q: MitsubaOptimizationType
    ) -> MitsubaOptimizationType:
        """
        Computes the inverse Hessian approximation H_k product with vector `q`
        for variable with key `variable_key` using the L-BFGS two-loop recursion.

        This method is used for finding the search direction in the L-BFGS
        optimization step. It applies the two-loop recursion algorithm to compute
        the product of the inverse Hessian approximation with the vector `q`,
        representing the current gradient.

        The Notation in the implementation follows the standard L-BFGS notation
        (e.g. Wikipedia: https://en.wikipedia.org/wiki/Limited-memory_BFGS
        or Nocedal and Wright's book on numerical optimization):

        - `s[i]` is the difference between the current and previous parameter
          state at iteration `i`
        - `y[i]` is the difference between the current and previous gradient
            at iteration `i`
        - `ys[i]` is the dot product of `y[i]` and `s[i]`
        - `q` is the current gradient
        - `r` is the produjct of `q` with the inverse Hessian approximation

        :returns: The result of the inverse Hessian approximation applied to `q`.
            Taking the negative of the result gives the search direction
            for the L-BFGS optimization step.
        """
        s_hist = self._state_difference[variable_key]
        y_hist = self._gradient_differences[variable_key]
        ys_hist = self._gradient_diff_dot_state_diff[variable_key]

        alphas: list[float] = []

        for s, y, ys in zip(reversed(s_hist), reversed(y_hist), reversed(ys_hist)):
            alpha = dr.rcp(ys) * dr.dot(s, q)
            q = q - alpha * y
            alphas.insert(0, alpha)

        gamma = (
            1.0
            if self._iterations[variable_key] == 1
            else ys_hist[-1] * dr.rcp(dr.dot(y_hist[-1], y_hist[-1]))
        )
        r = gamma * q

        for s, y, ys, alpha in zip(s_hist, y_hist, ys_hist, alphas):
            b = dr.rcp(ys) * dr.dot(y, r)
            r = r + (alpha - b) * s

        return r

    @staticmethod
    def _fill_new_values(
        old_params_dict: dict[str, MitsubaOptimizationType],
        new_params_dict: dict[str, MitsubaOptimizationType],
        alpha: dict[str, float],
        search_dirs: dict[str, MitsubaOptimizationType],
        enable_grad: bool = True,
    ) -> None:
        """
        Fills the `new_params_dict` with new values computed from the `old_params_dict`
        and `search_dirs` using the step size `alpha`.

        This method stores in `new_params_dict` the new values for each variable
        computed as `new_params = old_params + alpha * search_dirs`. It handles
        the logic for reshaping the search directions if they are tensors and
        ensures that the new parameters are detached from the computation graph
        unless `enable_grad` is set to True.

        :raises ValueError: If the keys of the input dictionaries do not match (this
            does not include the `new_params_dict` as it can be empty when passed to
            this method).
        """
        if (
            old_params_dict.keys() != alpha.keys()
            or old_params_dict.keys() != search_dirs.keys()
        ):
            raise ValueError(
                "All input dictionaries must have the same keys! Got "
                f"{old_params_dict.keys()}, {alpha.keys()}, {search_dirs.keys()}."
            )

        for variable_key, params in old_params_dict.items():
            if type(params) is mi.TensorXf:
                reshaped = dr.reshape(
                    dtype=mi.TensorXf,
                    value=search_dirs[variable_key],
                    shape=params.shape,
                )
                new_params_dict[variable_key] = dr.detach(
                    params + alpha[variable_key] * reshaped
                )
            else:
                new_params_dict[variable_key] = dr.detach(
                    params
                    + alpha[variable_key]
                    * dr.unravel(type(params), search_dirs[variable_key])
                )

            if enable_grad:
                dr.enable_grad(new_params_dict[variable_key])

            dr.schedule(new_params_dict[variable_key])

    @staticmethod
    def _backtracking_line_search(
        objective: MitsubaOptimizationType,
        previous_params: dict[str, MitsubaOptimizationType],
        previous_gradients: dict[str, MitsubaOptimizationType],
        search_directions: dict[str, MitsubaOptimizationType],
        initial_step_sizes: dict[str, float],
        use_wolfe_curvature: bool,
        objective_function: Callable[
            [dict[str, MitsubaOptimizationType]], MitsubaOptimizationType
        ],
        line_search_iterations: int = 20,
    ) -> dict[str, float]:
        """
        Perform backtracking line search to find an acceptable step size that
        satisfies the Armijo condition.

        Given the current `objective` value, `previous_params`,
        `previous_gradients`, `initial_step_sizes`, and the computed search
        `search_directions`, this method iteratively reduces the step size by half
        until the Armijo condition, and optionally the Wolfe curvature condition,
        is satisfied for all variables. These two conditions are defined as follows:

            1. Armijo condition: Ensures sufficient decrease in the objective function.
                f_new <= f_old + c1 * alpha * g_dot_z
            2. Wolfe curvature condition (if enabled): Ensures that the new gradient
                is not too steep compared to the old gradient.
                g_new_dot_z >= c2 * g_dot_z

        where f_new is the new objective value, f_old is the `objective` value,
        c1 and c2 are constants (armijo_const and wolfe_const), alpha is the step
        candidate size, g_dot_z is the dot product of the previous gradient and the
        search direction, g_new_dot_z is the dot product of the new gradient and the
        search direction. The new value (and gradient) is computed by calling the
        `objective_function` with the candidate parameters.

        The conditions have to be satisfied for all variables that the optimizer
        updates at each step.

        The method is written in a self-contained manner (staticmethod) as line search
        is not an inherent property of L-BFGS, but rather a general technique used in
        optimization problems. Right now it is not used outside L-BFGS so it is
        associated with this class.

        :returns: A dictionary of step sizes for each variable that satisfies the
            Armijo condition (and Wolfe conditions if enabled).

        :raises ValueError: If the keys of the input dictionaries do not match.
        """
        if (
            previous_params.keys() != previous_gradients.keys()
            or previous_params.keys() != search_directions.keys()
            or previous_params.keys() != initial_step_sizes.keys()
        ):
            raise ValueError(
                "All input dictionaries must have the same keys! Got "
                f"{previous_params.keys()}, {previous_gradients.keys()}, "
                f"{search_directions.keys()}, {initial_step_sizes.keys()}."
            )

        alpha = initial_step_sizes

        armijo_const: float = 1e-4
        """
        Armijo condition parameter for sufficient decrease. Default value from 
        Nocedal and Wright's book on numerical optimization.
        """
        wolfe_const: float = 0.9
        """
        Wolfe condition parameter for curvature condition. Default value from
        Nocedal and Wright's book on numerical optimization.
        """

        for it in range(line_search_iterations):
            candidate_params_dict: dict[str, MitsubaOptimizationType] = {}
            """
            Candidate new parameters after adding the search direction with the 
            current step size to the old parameters.
            """

            LBFGS._fill_new_values(
                old_params_dict=previous_params,
                new_params_dict=candidate_params_dict,
                alpha=alpha,
                search_dirs=search_directions,
                enable_grad=use_wolfe_curvature,
            )

            new_objective = objective_function(candidate_params_dict)
            """Objective value for the candidate parameters."""

            # Compute the directional derivative with respect to all variables.
            g_dot_z = mi.Float(0.0)
            alpha_g_dot_z = mi.Float(0.0)

            for variable_key, params in candidate_params_dict.items():
                g_dot_z_variable = dr.dot(
                    previous_gradients[variable_key], search_directions[variable_key]
                )

                alpha_g_dot_z += alpha[variable_key] * g_dot_z_variable
                g_dot_z += g_dot_z_variable

            armijo_condition = new_objective <= objective + armijo_const * alpha_g_dot_z
            condition_satisfied = armijo_condition

            # If Wolfe curvature condition is enabled, both the Armijo and Wolfe
            # condition have to be satisfied.
            if use_wolfe_curvature:
                # Wolfe curvature condition requires the new gradient to be computed
                dr.backward(new_objective)
                g_new_dot_z = mi.Float(0.0)

                for variable_key, params in candidate_params_dict.items():
                    g_new_dot_z += dr.dot(
                        dr.ravel(params.grad), search_directions[variable_key]
                    )

                wolfe_curvature_condition = g_new_dot_z >= wolfe_const * g_dot_z

                condition_satisfied &= wolfe_curvature_condition

            # Accept step size if the condition is satisfied
            if dr.all(condition_satisfied):
                logger.debug(
                    f"Line search iteration {it + 1} (Wolfe = {use_wolfe_curvature}) - "
                    f"f_new = {new_objective}, f_old = {objective}."
                )
                break

            for variable_key in alpha.keys():
                alpha[variable_key] *= 0.5

        return alpha

    def step(self, objective: MitsubaOptimizationType) -> None:
        """
        Perform a single L-BFGS optimization step.

        Computes search directions for all variables using the L-BFGS two-loop
        recursion algorithm, then performs backtracking line search to find
        an acceptable step size that satisfies the specified conditions.
        `objective` is the current value of the objective function

        The algorithm:
        1. Updates history for all variables
        2. Computes search directions using L-BFGS two-loop recursion
        3. Performs backtracking line search with Armijo (and Wolfe) conditions
        4. Updates all variables with the computed step

        Using Wolfe curvature condition requires the gradients to be computed
        for the new parameters after each line search iteration, which can be
        computationally expensive.

        :raises RuntimeError: If line_search_fn is not set.
        :raises ValueError: If no variables are registered for optimization.
        """
        search_directions: dict[str, MitsubaOptimizationType] = {}
        """Search directions for each variable, computed using the L-BFGS algorithm."""

        # Compute search directions
        for variable_key, params in self.variables.items():
            if variable_key not in self._state_difference:
                self.reset(variable_key)

            # Reset history if the shape of the parameter has changed
            if (
                self._iterations[variable_key] > 0
                and self._previous_params[variable_key] is not None
                and params.shape != self._previous_params[variable_key].shape
            ):
                self.reset(variable_key)

            # Update history
            self._update_history(variable_key)

            search_directions[variable_key] = -1.0 * self._apply_inverse_hessian(
                variable_key=variable_key, q=dr.ravel(params.grad)
            )

        if self._line_search_fn is None:
            raise RuntimeError(
                "Line search function is not set! Please set the `line_search_fn` "
                "property before calling `step`."
            )

        # Backtracking line search until Armijo (and optionally Wolfe) conditions are
        # satisfied or the maximum number of iterations is reached.
        alpha = self._backtracking_line_search(
            objective=objective,
            previous_params=self.variables,
            previous_gradients=self._previous_gradient,
            search_directions=search_directions,
            initial_step_sizes=self.lr_v,
            use_wolfe_curvature=self._use_wolfe_curvature,
            objective_function=self._line_search_fn,
            line_search_iterations=self._line_search_iterations,
        )

        self._fill_new_values(
            old_params_dict=self.variables,
            new_params_dict=self.variables,
            alpha=alpha,
            search_dirs=search_directions,
            enable_grad=True,
        )

    @property
    def line_search_fn(
        self,
    ) -> Callable[[dict[str, MitsubaOptimizationType]], MitsubaOptimizationType]:
        """
        Returns the objective function used for line search.

        The line search function takes in a dictionary that contains all the variables
        that are being optimized and returns the total loss for the input parameters.
        These intput parameters are the new proposed values that have to be checked
        with line search to see if they satisfy the Armijo (and Wolfe) conditions.

        This function should be the same as the one used to compute the loss during
        the optimization step. The seed for the random number generator should also
        be the same.

        :returns: A callable function that takes a dictionary of variables and returns
            the total loss for the input parameters.
        """
        return self._line_search_fn

    @line_search_fn.setter
    def line_search_fn(
        self,
        new_line_search_fn: Callable[
            [dict[str, MitsubaOptimizationType]], MitsubaOptimizationType
        ],
    ) -> None:
        """
        Sets the objective function used for line search.

        The function should take a dictionary of variables as input and return the
        total loss for the input parameters. It should be the same as the one used
        to compute the loss during the optimization step. The seed for the random
        number generator should also be the same.

        The function is called during the line search to compute the new objective
        value for the candidate parameters.
        """
        self._line_search_fn = new_line_search_fn
