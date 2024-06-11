import logging
from pathlib import Path
import numpy as np
import open3d as o3d
from lydar.utils import load_json

class PCLydar:
    def __init__(self, dirname):
        self.fname = dirname / "point_cloud.ply"
        fname = str(self.fname)
        
        if self.fname.is_file():
            log.debug(f"File {fname!r} exists, skipping")
        else:
            lidars = load_json(dirname / "lidars.json")
            inter_pts = load_json(dirname / "intersection_points.json")
            pts = []
            
            for lid in lidars:
                for pt in inter_pts[lid["name"]]:
                    pts.append(pt['x'])
            
            # Convert the list to a NumPy array
            points_np = np.array(pts)

            # Create an Open3D PointCloud object
            pcd = o3d.geometry.PointCloud()

            # Set the points of the PointCloud object
            pcd.points = o3d.utility.Vector3dVector(points_np)

            # Save the point cloud to a .ply file
            o3d.io.write_point_cloud(fname, pcd)
            print(f"Point cloud saved to {fname}")

    def plot(self):
        pcd = o3d.io.read_point_cloud(str(self.fname))
        o3d.visualization.draw_geometries([pcd])
    
    def get_outliers(self):
        pcd = o3d.io.read_point_cloud(str(self.fname))
        
        def ransac(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=100):
            plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
            inlier_cloud = pcd.select_by_index(inliers)
            outlier_cloud = pcd.select_by_index(inliers,invert=True)
            inlier_cloud.paint_uniform_color([1,0,0])
            outlier_cloud.paint_uniform_color([0,0,1])  
            return inlier_cloud, outlier_cloud

        inlier_cloud, outlier_cloud = ransac(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=100)
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        return outlier_cloud
    
    def segment(self, pcd):
        # pcd = o3d.io.read_point_cloud(str(self.fname))
        
        labels = np.array(pcd.cluster_dbscan(eps=1.5, min_points=50))
        print(labels, len(labels), np.unique(labels))

        for x in np.unique(labels):
            object_cluster = pcd.select_by_index(np.where(labels == x)[0])  # Adjust cluster_id as needed
            print(object_cluster)
            o3d.visualization.draw_geometries([object_cluster])

        # # Estimate normals
        # object_cluster.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # # Surface reconstruction
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(object_cluster, depth=9)
        # mesh = mesh.filter_smooth_simple(number_of_iterations=5)

        # # Save the mesh
        # o3d.io.write_triangle_mesh("reconstructed_object.ply", mesh)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    root = Path(__file__).parent.parent.parent / "out"
    pc = PCLydar(root)
    # pc.plot()
    outlier_cloud = pc.get_outliers()
    pc.segment(outlier_cloud)
