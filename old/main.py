from collections import defaultdict
from pprint import pprint
import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool, cpu_count

pv.global_theme.allow_empty_mesh = True


class Scene:
    def __init__(self):
        self.floor = pv.Plane(center=(0, 0, 0), direction=(
            0, 0, 1), i_size=100, j_size=100)

        self.objects = {'floor': self.floor}

        # self.rot = R.from_rotvec(np.pi/4 * np.array([1, 0, 0]))
        self.rot = R.from_rotvec(np.deg2rad(20) * np.array([1, 0, 0]))


# Créer un environnement virtuel
def create_environment(scene):
    plotter = pv.Plotter()

    # Créer un plan pour le sol
    plotter.add_mesh(scene.floor, color='tan', opacity=0.5)

    # Créer un cube à côté du sol
    cube = pv.Cube(center=(3, 0, 0.5), x_length=1, y_length=1, z_length=1)
    plotter.add_mesh(cube, color='green', opacity=0.5)
    scene.objects['cube_1'] = cube

    cube2 = pv.Cube(center=(6, -1, 2), x_length=1, y_length=1, z_length=4)
    plotter.add_mesh(cube2, color='green', opacity=0.5)
    scene.objects['cube_2'] = cube2

    return plotter

# Placer des points à des coordonnées données


def place_points(plotter, points):
    for point in points:
        plotter.add_mesh(pv.Sphere(radius=0.1, center=point), color='red')

# Définir la zone d'influence


def worker(objs, ray_points):
    inters = []
    for tag, obj in objs.items():
        intersections = obj.ray_trace(ray_points[0], ray_points[1])
        for intersection in intersections:
            if len(intersection) > 0:
                intersection_point = intersection[0]
                # plotter.add_mesh(pv.Sphere(radius=0.1, center=intersection_point), color='green')
                inters.append(
                    {"x": intersection_point, "dist": np.linalg.norm(intersection_point-ray_points[0]), 'tag': tag})
                break
    if len(inters) > 0:
        inter_p = sorted(inters, key=lambda x: x['dist'])[0]
        return inter_p
        # points[inter_p['tag']].append(inter_p['x'])
        # plotter.add_mesh(
        #     pv.Sphere(radius=0.1, center=inter_p['x']), color='green')


def define_zone(scene, plotter, point, v_angle=40, h_res=1, v_res=1):
    h_steps = int(360/h_res)
    v_steps = int(v_angle/v_res)

    # Angle azimutal int(360 / step)
    theta = np.linspace(0, 2 * np.pi, h_steps)
    xx = 0.5 * np.radians(v_angle)
    pi2 = np.pi*0.5
    # Angle d'élévation int(v_angle / step)
    phi = np.linspace(pi2-xx, pi2+xx, v_steps)

    ll = 200
    points = defaultdict(list)
    workers_data = []
    for t in theta:
        for p in phi:
            x = point[0] + ll * np.sin(p) * np.cos(t)
            y = point[1] + ll * np.sin(p) * np.sin(t)
            z = point[2] + ll * np.cos(p)

            rot_point = scene.rot.apply(np.array([x, y, z]))
            # line = pv.Line(point, rot_point)
            # plotter.add_mesh(line, color='blue', line_width=1)

            workers_data.append((scene.objects, (point, rot_point)))

            # # Utilisation de ray_trace pour trouver les intersections avec le plan
            # inters = []
            # for tag, obj in scene.objects.items():
            #     intersections = obj.ray_trace(point, rot_point)
            #     for intersection in intersections:
            #         if len(intersection) > 0:
            #             intersection_point = intersection[0]
            #             # plotter.add_mesh(pv.Sphere(radius=0.1, center=intersection_point), color='green')
            #             inters.append(
            #                 {"x": intersection_point, "dist": np.linalg.norm(intersection_point-point), 'tag': tag})
            #             break
            # if len(inters) > 0:
            #     inter_p = sorted(inters, key=lambda x: x['dist'])[0]
            #     points[inter_p['tag']].append(inter_p['x'])
            #     plotter.add_mesh(
            #         pv.Sphere(radius=0.1, center=inter_p['x']), color='green')

    print("DEBUT PARA")
    with Pool(cpu_count()) as p:  # cpu_count()=12
        points_ = list(p.starmap(worker, workers_data))
    print("DEBUT FIN")

    for x in points_:
        if x is not None:
            points[x['tag']].append(x['x'])
            if x['tag'] == 'floor':
                plotter.add_mesh(
                    pv.Sphere(radius=0.075, center=x['x']), color='green')

    return points


def add_delaunay_surface(points, plotter, **kwargs):
    """
    Add a Delaunay surface reconstructed from a list of points to a PyVista plotter.

    Parameters:
        points (list of lists or numpy.ndarray): List of 2D points [[x1, y1], [x2, y2], ...].
        plotter (pyvista.Plotter): PyVista plotter object.
        **kwargs: Additional keyword arguments to pass to the add_mesh function.

    Returns:
        surface: PyVista mesh representing the reconstructed surface.
    """
    # Create a PyVista PolyData object from the input points
    polydata = pv.PolyData(points)

    # Perform Delaunay 2D triangulation
    surf = polydata.delaunay_2d()

    # Add the reconstructed surface to the plotter
    surface = plotter.add_mesh(surf, color='red', show_edges=True, **kwargs)

    return surface


def main():
    scene = Scene()

    # Points à placer
    points = [
        (1, 1, 1),
        (-1, -1, 0.5),
        # (2, -2, 0.5),
    ]

    # Créer l'environnement
    plotter = create_environment(scene)

    # Placer les points
    place_points(plotter, points)

    # Définir les zones d'influence pour chaque point
    for point in points:
        points_zone = define_zone(scene, plotter, point, v_angle=40, h_res=1.5, v_res=2)

        # add_delaunay_surface(points_zone['floor'], plotter)
        add_delaunay_surface(points_zone['cube_1'], plotter)
        add_delaunay_surface(points_zone['cube_2'], plotter)

    # Afficher la scène
    plotter.show()


if __name__ == "__main__":
    main()
