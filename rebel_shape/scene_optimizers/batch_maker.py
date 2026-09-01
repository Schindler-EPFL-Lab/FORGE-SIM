import numpy as np


class BatchMaker:
    """
    Utility for partitioning and shuffling dataset indices into balanced batches.

    This class provides utilities to divide a total number of items (such as reference
    images) into balanced batches that do not exceed a specified maximum size. It also
    supports deterministic shuffling of item indices and batch sequences for
    reproducible optimization iterations.
    """

    def __init__(
        self,
        max_batch_size: int | None = None,
        randomize_image_order: bool = True,
        image_order_seed: int = 42,
    ) -> None:
        """
        Initializes the batch maker with batch sizing and randomization settings

        `max_batch_size` is the maximum number of images that can be used for gradient
        update in a single shape update during an optimization step (effectively each
        step is an epoch over all images).

        At optimization time, if the number of images in `ref_images` is
        larger than `max_batch_size`, multiple shape updates are performed in
        `run_optimization_step`, each considering a subset of images. This limits
        memory consumption and is especially useful for optimizers like Adam.
        Setting it to `None` means that all images are used for each update.

        `randomize_image_order` controls whether the order of images is shuffled
        before each optimization step. This can help improve convergence by reducing
        correlations between images. The shuffling is done using a fixed seed
        (`image_order_seed`) to ensure reproducibility. As it is possible for batches
        to have different sizes, the order of batches is also shuffled.
        """
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("`max_batch_size` has to be a positive integer")

        self._max_batch_size = max_batch_size
        self._randomize_image_order = randomize_image_order
        self._image_order_seed = image_order_seed

        self._rng = np.random.default_rng(image_order_seed)

    def _batch_sizes(self, num_images: int) -> list[int]:
        """
        Compute batch sizes to split `num_images` into batches whose sizes differ by at
        most 1 and do not exceed `self._max_batch_size`.

        Behavior:
        - If `self._max_batch_size` >= `num_images`, return a single batch.
        - Otherwise choose the minimal number of batches that do not exceed
          `self._max_batch_size` and distribute `num_images` as evenly as possible
          across those batches. The first `remainder` batches will have size `base +
          1`, the rest `base`.

        Examples:
        - num_images=20, max_batch_size=50 -> num_batches=1 -> [20]
        - num_images=119, max_batch_size=50 -> num_batches=3 -> [40, 40, 39]
        - num_images=118, max_batch_size=50 -> num_batches=3 -> [39, 40, 39]

        :returns: List of batch sizes summing to `num_images`.
        :raises ValueError: if `num_images` is less than 1.
        """
        if num_images < 1:
            raise ValueError("`num_images` must be at least 1.")

        if self._max_batch_size is None or self._max_batch_size >= num_images:
            return [num_images]

        # Compute the minimal number of batches such that no batch exceeds
        # `self._max_batch_size`, i.e. num_batches * max_batch_size >= num_images
        num_batches = (num_images + self._max_batch_size - 1) // self._max_batch_size

        # It is always possible to split num_images into num_batches such that
        # the batch sizes differ by at most 1.
        base = num_images // num_batches
        remainder = num_images % num_batches

        return [base + 1] * remainder + [base] * (num_batches - remainder)

    def iteration_batch_indices(self, num_images: int) -> list[list[int]]:
        """
        Compute the indices of images for each batch in an iteration.

        First, the number of batches and their sizes are determined using
        `self._batch_sizes`. All batches differ by at most 1 in size. Then, the indices
        of images are computed for each batch.

        If `self._randomize_image_order` is `True`, the order of images is shuffled
        before computing the batch indices. The shuffling is done using a fixed seed
        (`self._image_order_seed`) to ensure reproducibility.

        Furthermore, if `self._randomize_image_order` is `True`, the order of batches
        is also shuffled to avoid possible bias when batches have different sizes.

        :returns: A list of lists, where each inner list contains the indices of
            images for a batch.
        """
        batch_sizes = self._batch_sizes(num_images)
        indices = np.arange(num_images, dtype=np.uint32)

        # Randomize the order of images and batch sizes if enabled
        if self._randomize_image_order:
            self._rng.shuffle(indices)
            batch_sizes = self._rng.permutation(batch_sizes)

        batch_indices: list[list[int]] = []
        start = 0

        for size in batch_sizes:
            end = start + size
            batch_indices.append(indices[start:end].tolist())
            start = end

        return batch_indices
