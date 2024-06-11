import logging
from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from lydar.utils import DATA_PATH


class SceneObject:
    TYPE = "SceneObject"

    def __init__(self, name):
        self.name = name

    def add_to(self, plotter):
        raise NotImplementedError


class Mesh(SceneObject):
    TYPE = "MESH"

    def __init__(self, name, mesh, **kwargs):
        super().__init__(name)
        self.pvmesh = mesh
        self.kwargs = kwargs

    def add_to(self, plotter):
        plotter.add_mesh(self.pvmesh, **self.kwargs)


class Lidar(SceneObject):
    TYPE = "LIDAR"

    def __init__(self, name, position, rotation: Rotation, h_res=5, v_res=5, v_angle=40, color='k'):
        super().__init__(name)
        self.position: np.ndarray = np.array(position)
        self.rotation: Rotation = rotation
        self.h_res: float = h_res
        self.v_res: float = v_res
        self.v_angle: float = v_angle
        self.color = color

    def add_to(self, plotter):
        # plotter.add_mesh(
        #     pv.Sphere(radius=0.1, center=self.position), color=self.color)

        mesh = pv.read(DATA_PATH / "lidar.stl").scale(1e-3)
        center = mesh.points[464] - np.array([0,0,-0.045])
        
        M = np.eye(4)
        M[:3,:3] = self.rotation.as_matrix()
        M[:3, 3] = self.position - center
        mesh = mesh.rotate_x(90).rotate_z(90).transform(M).scale(2)
        
        plotter.add_mesh(mesh, color=self.color)

    def to_dict(self):
        return {
            "name": self.name,
            "position": self.position,
            "rotation": self.rotation.as_matrix(),
            "h_res": self.h_res,
            "v_res": self.v_res,
            "v_angle": self.v_angle,
            "color": self.color,
        }


log = logging.getLogger(__name__)
