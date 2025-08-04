import abc
from typing import Any, Union

import mitsuba as mi


class MitsubaPlugin(abc.ABC):
    """
    An interface that is meant to ease communication with Mitsuba 3.

    Implementing classes return a Mitsuba compatible dictionary that
    is used to construct scenes while exposing type hints and options on
    the Python side. For more information about Mitsuba plugins visit
    https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html
    """

    @property
    @abc.abstractmethod
    def mitsuba_dict(self) -> Union[dict[str, Any], "mi.Mesh"]:
        """
        Provides a Mitsuba compatible dictionary representation of the class

        :returns: A dictionary with keys and values prescribed by its respective
            plugin reference.
        :raises NotImplementedError: if the concrete implementation has not
            implemented this property
        """
        raise NotImplementedError(
            "Classes that inherit from `MitsubaPlugin` need to implement the "
            "`mitsuba_dict` property."
        )
