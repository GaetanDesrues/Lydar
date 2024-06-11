import logging
from pathlib import Path
from pprint import pprint

from scipy.spatial.transform import Rotation as R
import numpy as np
import pyvista as pv

from lydar.objects import Lidar, Mesh
from lydar.scene import LydarScene
from lydar.utils import DATA_PATH


def main():
    root = Path(__file__).parent / "out"
    root.mkdir(exist_ok=True)

    show = False
    scene = LydarScene(root, off_screen=not show)
    lidar_res = {'h_res': 5, 'v_res': 5}
    # lidar_res = {'h_res': 0.5, 'v_res': 1}

    lidar_1 = Lidar('lidar_1', [1, 1, 1], R.from_rotvec(
        np.deg2rad(20) * Y), color='g', **lidar_res)
    scene.add(lidar_1)

    lidar_2 = Lidar('lidar_2', [-1, 1, 1],
                    R.from_rotvec(np.deg2rad(-20) * Y), color='b', **lidar_res)
    scene.add(lidar_2)

    floor = pv.Plane(center=(0, 0, 0), direction=(
        0, 0, 1), i_size=50, j_size=50)
    scene.add(Mesh('floor', floor, color='tan', opacity=0.5))

    cube_1 = pv.Cube(center=(-5, -5, 0.5), x_length=1, y_length=1, z_length=1)
    scene.add(Mesh('cube_1', cube_1, color='w', opacity=0.5))

    cube_2 = pv.Cube(center=(10, -15, 2), x_length=1, y_length=1, z_length=4)
    scene.add(Mesh('cube_2', cube_2, color='w', opacity=0.5))

    raptor = pv.read(DATA_PATH / "raptor.obj")
    raptor = raptor.rotate_x(90).rotate_z(180).translate([5, -10, 0])
    scene.add(Mesh('raptor', raptor, color='w', opacity=1))

    print("Scene objects:")
    pprint(scene.scene_objects)

    scene.compute_frame()

    cam = [(-7.659618450627384, 21.569617083479073, 9.616754046194163),
           (-0.18574919177545401, -0.6669005126682008, 2.276331169659448),
           (0.09210181616950845, -0.2840860744904501, 0.9543648975831437)]
    scene.plotter.camera_position = cam

    def take_screenshot():
        scene.plotter.screenshot(root / "pts.png")
        scene.plotter.close()

    scene.plotter.add_callback(take_screenshot, 1000, 1)
    scene.render(show=show)


X, Y, Z = np.eye(3).T
log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    main()
