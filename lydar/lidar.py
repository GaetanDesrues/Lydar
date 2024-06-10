import numpy as np
import randomname


class Lidar:
    def __init__(self, position, rotation, name=randomname.get_name()):
        self.position = np.array(position)
        self.rotation = np.array(rotation)
        self.name = name
