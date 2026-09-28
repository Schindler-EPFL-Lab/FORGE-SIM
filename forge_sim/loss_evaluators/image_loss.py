import abc

import mitsuba as mi


class ImageLoss(abc.ABC):
    """
    `drjit` compatible loss function.

    Dr.Jit (`drjit`) is a library that Mitsuba uses to perform automatic
    differentiation. Since external code must be able to call `dr.backward()`
    all types that are used here have to be `drjit` compatible. They can easily
    be converted to numpy or pytorch if needed (consult the docs or tutorial).
    """

    @abc.abstractmethod
    def evaluate(self, image: "mi.TensorXf", ref_image: "mi.TensorXf") -> "mi.TensorXf":
        """
        Computes the loss for a single `image`. The loss is strictly positive and
        should return 0 if `ref_image` and `image` match perfectly (i.e. if
        a reference image is passed to both parameters).

        :returns: `drjit` tensor with a single element. To extract the floating point
            value from the tensor use `.numpy().astype(np.float64)`.

        :raises NotImplementedError: if the concrete implementation has not
            implemented this property
        """
        raise NotImplementedError(
            "Classes that inherit from `ImageLoss` need to implement the "
            "`evaluate` method."
        )
