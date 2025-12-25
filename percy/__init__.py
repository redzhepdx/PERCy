from __future__ import annotations

from typing import Any, Optional

import importlib
_cper = importlib.import_module("percy.per")


class PyPER:
    def __init__(self, capacity : int=256, alpha : float=0.6, beta : float=0.4) -> None:
        """"
        Initializes a Prioritized Experience Replay (PER) buffer.
        Args:
            capacity (int): Maximum number of items the buffer can hold.
            alpha (float): Degree of prioritization (0 - no prioritization, 1 - full prioritization).
            beta (float): Degree of importance-sampling correction (0 - no correction, 1 - full correction).
        """
        self.per_obj = _cper.create(capacity=capacity, alpha=alpha, beta=beta)

    def add(self, item : Any, priority : Optional[float]=None) -> None:
        """"
        Adds an item to the PER buffer with an optional priority.
        Args:
            item (Any): The item to be added to the buffer.
            priority (Optional[float]): The priority of the item. If None, default priority is used.
        """
        if priority is not None:
            _cper.add(self.per_obj, item, priority)
        else:
            _cper.add(self.per_obj, item)

    def sample(self, batch_size : int) -> list[Any]:
        """
        Samples a batch of items from the PER buffer.
        Args:
            batch_size (int): Number of items to sample.
        Returns:
            list[Any]: A list of sampled items.
        """
        return _cper.sample(self.per_obj, batch_size)

    def update_priorities(self, p_indices : list[int], td_errors : list[float]) -> None:
        """
        Updates the priorities of items in the PER buffer.
        Args:
            p_indices (list[int]): List of indices of the items to update.
            td_errors (list[float]): List of new priority values (TD-errors) for the items.
        """
        _cper.update_priorities(self.per_obj, p_indices, td_errors)

    def __len__(self) -> int:
        """
        Returns the current number of items in the PER buffer.
        Returns:
            int: Number of items in the buffer.
        """
        return _cper.size(self.per_obj)
    
    @property
    def total_priority(self) -> float:
        """
        Returns the total priority of all items in the PER buffer.
        Returns:
            float: Total priority.
        """
        return _cper.total_priority(self.per_obj)

    def close(self):
        # optional cleanup if your C extension exposes a free method
        self.per_obj = None