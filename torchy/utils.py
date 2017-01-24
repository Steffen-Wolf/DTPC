from contextlib import contextmanager
from argparse import Namespace
import time


@contextmanager
def timeit():
    """
    Context manager that times what happens under it. Maybe with a wee-bit overhead,
    but shouldn't matter much for relative measurements.

    Usage:
    ```
    with timeit() as timeit_info:
        output = ...

    print("Elapsed time: {}".format(timit_info.elapsed_time))
    ```
    """
    timeit_info = Namespace()
    now = time.time()
    yield timeit_info
    later = time.time()
    timeit_info.elapsed_time = later - now


def simulate_delay(duration):
    """Delay a given function by a certain `duration` (in seconds)."""
    def _decorator(function):
        def _function(*args, **kwargs):
            time.sleep(duration)
            return function(*args, **kwargs)
        return _function
    return _decorator
