import logging

from scipy.spatial.transform import Rotation as R
import numpy as np

from lydar.lidar import Lidar
from lydar.scene import LydarScene


X, Y, Z = np.eye(3).T


def main():
    scene = LydarScene()

    lidar_1 = Lidar([1, 1, 1], R.from_rotvec(np.deg2rad(20) * X))
    scene.add_lidar(lidar_1)
    
    scene.start()

    # sphere = pv.Sphere()

    # plotter = BackgroundPlotter(show=False)
    # plotter.add_mesh(sphere)

    # plotter.app_window.show()
    # plotter.app.exec_()


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
