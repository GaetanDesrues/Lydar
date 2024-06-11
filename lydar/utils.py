import logging
from pathlib import Path, PosixPath
import time
import numpy as np
import json
import pyvista as pv


DATA_PATH = Path(__file__).parent / "data"

pv.global_theme.allow_empty_mesh = True


def dump_json(filename: str, data: dict):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=CustomEncoder, indent=4)


class CustomEncoder(json.JSONEncoder):
    def default(self, z):
        from lydar.objects import SceneObject

        if isinstance(z, np.ndarray):
            return z.tolist()
        elif isinstance(z, SceneObject):
            return z.to_dict()
        elif isinstance(z, PosixPath):
            return str(z.absolute())
        else:
            return super().default(z)


class Timer:
    """
    Basic timer with context

    .. code-block::

        with Timer() as t:
            time.sleep(1.46)
        print(t.secs)  # 1.460746487020515

    """

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.secs = end_time - self.start_time  # seconds


log = logging.getLogger(__name__)
