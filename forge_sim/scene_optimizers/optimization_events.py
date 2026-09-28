from typing import Callable


class OptimizationEvents:
    """
    Represents a collection of all events that should occur during optimization,
    at each iteration.

    Each event is uniquely defined by the iteration it happens in and a callback
    that should not take any inputs not produce outputs. If multiple events are
    triggered in the same iteration they should be executed in the order they were
    added with the `add_event` method.
    """

    def __init__(self) -> None:
        """
        Initializes an empty collection of optimization events.
        """
        self._events: dict[int, list[Callable[[], None]]] = {}

    def add_event(self, iteration: int, callback: Callable[[], None]) -> None:
        """
        Adds a new event to be triggered at the specified `iteration`.

        The even is defined by a callback function that should not take any inputs
        and should not produce any outputs. The `callback` is executed at the end of
        the iteration.

        :raises ValueError: If `iteration` is negative.
        """
        if iteration < 0:
            raise ValueError("Iteration must be a non-negative integer.")

        if iteration not in self._events:
            self._events[iteration] = []
        self._events[iteration].append(callback)

    def trigger_events(self, iteration: int) -> None:
        """
        Triggers all events registered for the given iteration.

        Events are executed in the order they were added.
        """
        if iteration in self._events:
            for callback in self._events[iteration]:
                callback()

    def clear_events(self) -> None:
        """
        Removes all registered events.
        """
        self._events = {}