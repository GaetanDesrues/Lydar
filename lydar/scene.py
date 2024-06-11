from collections import defaultdict
import logging
from functools import partial
from pprint import pprint
from typing import Dict
import pyvista as pv
import numpy as np
import multiprocessing as mp

from lydar.objects import Lidar, Mesh, SceneObject
from lydar.plotter import LydarPlotter
from lydar.utils import Timer, dump_json


class LydarScene:
    def __init__(self, root, off_screen=False):
        self.root = root  # output dir
        self.plotter = LydarPlotter(
            off_screen=off_screen,
            # window_size=(800, 500),
        )
        self.scene_objects: Dict[str, SceneObject] = {}

    @property
    def lidars(self):
        return [x for x in self.scene_objects.values() if x.TYPE == "LIDAR"]

    @property
    def meshes(self):
        return [x for x in self.scene_objects.values() if x.TYPE == "MESH"]

    def add(self, obj: SceneObject, add_to_plotter=True):
        self.scene_objects[obj.name] = obj
        if add_to_plotter:
            obj.add_to(self.plotter)

    def compute_frame(self):
        """For the current scene_objects, compute ray tracing for each lidar and its FOV to each object"""

        # For each lidar:
        meshes, lidars = self.meshes, self.lidars
        intersections = {}
        for lid in lidars:
            with Timer() as t:
                pts = get_points(lid)

                # For each object in the scene (parallel for each object)
                worker = partial(
                    intersect, origin=lid.position, ray_points=pts)
                # with mp.Pool(min(mp.cpu_count(), len(meshes))) as p:
                #     pts_intersect_objects = list(p.map(worker, meshes))
                pts_intersect_objects = [worker(x) for x in meshes]

                # If a ray intersected with several objects, keep the closest point
                pts_intersect = []
                for i in range(len(pts)):
                    found = [x[i] for x in pts_intersect_objects if i in x]
                    if len(found) > 0:
                        inter_p = sorted(found, key=lambda x: x['dist'])[0]
                        pts_intersect.append(inter_p)

                intersections[lid.name] = pts_intersect

            log.debug(
                f"{len(pts_intersect)} points found for {lid.name} in {t.secs:.2f}s")

        # pprint(intersections)
        dump_json(self.root / "intersection_points.json", intersections)
        dump_json(self.root / "lidars.json", lidars)

        # Display intersection points
        log.debug(f"Starting to plot intersection points")
        for lid, pts_obj in intersections.items():
            c = self.scene_objects[lid].color
            pts_ = np.array([pt['x'] for pt in pts_obj])
            self.plotter.add_points(
                pts_, point_size=15, color=c, render_points_as_spheres=True)
            # for pt in pts_obj:
            #     self.plotter.add_mesh(
            #         pv.Sphere(radius=0.1, center=pt['x']), color=c)
        log.debug(f"Finished to plot intersection points")

        # # For each mesh, reconstruct surface from intersected points
        # Use pv.reconstruct_surface (https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydatafilters.reconstruct_surface)
        # with Timer() as t:
        #     for lid, pts_obj in intersections.items():
        #         c = self.scene_objects[lid].color
        #         pts = defaultdict(list)
        #         for pt in pts_obj:
        #             if pt['tag'] != 'floor':
        #                 pts[pt['tag']].append(pt['x'])
        #         for k, v in pts.items():
        #             # polydata = pv.PolyData(v)
        #             # surf = polydata.delaunay_2d()
        #             # self.plotter.add_mesh(surf, color=c, show_edges=False)
        #             log.debug(f"Lidar {lid} captured the object {k!r}")
        #     log.debug(f"Reconstructed surfaces in {t.secs:.4f}s")

    def render(self, show=True):
        log.info(f"Loading visualisation...")

        with Timer() as t:
            self.plotter.start(show)
        log.info(f"Loaded visu in {t.secs:.2f}s")


def intersect(obj: Mesh, origin=None, ray_points=None):
    log.debug(f"Intersecting obj {obj.name}")
    inters = {}
    for i, pt in enumerate(ray_points):
        intersections = obj.pvmesh.ray_trace(origin, pt)
        for intersection in intersections:
            if len(intersection) > 0:
                intersection_point = intersection[0]
                inters[i] = {"x": intersection_point, "dist": np.linalg.norm(
                    intersection_point-origin), 'tag': obj.name}
                break
    return inters


def get_points(lidar: Lidar, max_lenght=200):
    """Prepare all ray-tracing points from the lidar"""
    log.info(f"Get points for lidar {lidar.name}")

    pos = lidar.position
    h_steps = int(360 / lidar.h_res)
    v_steps = int(lidar.v_angle / lidar.v_res)

    # Angle azimutal int(360 / res)
    theta = np.linspace(0, 2 * np.pi, h_steps)
    # Angle d'élévation int(v_angle / res)
    v2 = 0.5 * np.deg2rad(lidar.v_angle)
    phi = np.linspace(0.5*np.pi-v2, 0.5*np.pi+v2, v_steps)

    points = []
    for t in theta:
        for p in phi:
            x = pos[0] + max_lenght * np.sin(p) * np.cos(t)
            y = pos[1] + max_lenght * np.sin(p) * np.sin(t)
            z = pos[2] + max_lenght * np.cos(p)

            rot_point = lidar.rotation.apply(np.array([x, y, z]))
            points.append(rot_point)
    return points


log = logging.getLogger(__name__)
