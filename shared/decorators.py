import functools
import logging
import time
from typing import Any, Callable


def timer(function_to_measure: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to measure the execution time of a function.
    """
    _logger = logging.getLogger(function_to_measure.__name__)

    @functools.wraps(function_to_measure)
    def wrapper_timer(*args, **kwargs):
        print(f"Starting {function_to_measure.__name__}")
        _logger.info(f"Starting {function_to_measure.__name__}")
        start_time = time.perf_counter()
        value = function_to_measure(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {function_to_measure.__name__} in {run_time:.4f} seconds")
        _logger.info(f"Finished {function_to_measure.__name__} in {run_time:.4f} secs")
        return value

    return wrapper_timer
