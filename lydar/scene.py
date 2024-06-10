from typing import List
from lydar.lidar import Lidar
from lydar.plotter import LydarPlotter


class LydarScene:
    def __init__(self):
        self.plotter = LydarPlotter(show=False)
        self.lidars: List[Lidar] = []

    def add_lidar(self, lidar: Lidar):
        self.lidars.append(lidar)

    def start(self):
        self.plotter.start()
