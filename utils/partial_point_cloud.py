import open3d as o3d
import numpy as np
from tqdm import tqdm


def get_partial_point_clouds(mesh, view_vecs, num_points, visible_visualizer=False, use_tqdm=False, check_partial_pc=False):
    # Check open3d version
    assert o3d.__version__.split('.')[0] == '0' and o3d.__version__.split('.')[1] == '16', \
        f"open3d version must be 0.16, 'ctr.convert_from_pinhole_camera_parameters(camera_params)' doesn't work well in later versions"

    # Set distance from object center to camera
    distance = 1.5 * np.linalg.norm(mesh.get_oriented_bounding_box().extent)

    # Set visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visible_visualizer)

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()

    # Add mesh
    vis.add_geometry(mesh)

    # Set camera poses
    view_unit_vecs = view_vecs / np.linalg.norm(view_vecs, axis=1, keepdims=True)

    cam_z_s = - view_unit_vecs

    while True:
        cam_x_s = -1 + 2 * np.random.rand(len(cam_z_s), 3)
        cam_x_s = cam_x_s - np.sum(cam_x_s*cam_z_s, axis=1, keepdims=True) * cam_z_s

        if np.linalg.norm(cam_x_s, axis=1).any() == 0:
            continue
        else:
            cam_x_s /= np.linalg.norm(cam_x_s, axis=1, keepdims=True)

            break

    cam_y_s = np.cross(cam_z_s, cam_x_s)
    cam_y_s /= np.linalg.norm(cam_y_s, axis=1, keepdims=True)

    cam_Ts = np.tile(np.eye(4), (len(view_vecs), 1, 1))
    cam_Ts[:, :3, :3] = np.stack([cam_x_s, cam_y_s, cam_z_s], axis=2)
    cam_Ts[:, :3, 3] = distance * view_unit_vecs

    # Get partial point clouds
    partial_pcds = []

    if use_tqdm:
        pbar = tqdm(cam_Ts, desc="Iterating viewpoints ...", leave=False)
    else:
        pbar = cam_Ts

    for cam_T in pbar:
        # Set camera extrinsic parameters
        camera_params.extrinsic = np.linalg.inv(cam_T)

        ctr.convert_from_pinhole_camera_parameters(camera_params)

        # Update visualizer
        vis.poll_events()
        vis.update_renderer()

        # Get partial point cloud
        depth = vis.capture_depth_float_buffer()

        partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera_params.intrinsic, camera_params.extrinsic)

        # Raise Exception if the number of points in point cloud is less than 'num_points'
        if len(np.asarray(partial_pcd.points)) < num_points:
            raise Exception("Point cloud has an insufficient number of points. Increase visualizer window width and height.")

        # Downsample point cloud to match the number of points with 'num_points'
        else:
            voxel_size = 0.5
            voxel_size_min = 0
            voxel_size_max = 1

            while True:
                partial_pcd_tmp = partial_pcd.voxel_down_sample(voxel_size)

                num_points_tmp = len(np.asarray(partial_pcd_tmp.points))

                if num_points_tmp - num_points >= 0 and num_points_tmp - num_points < 100:
                    break
                else:
                    if num_points_tmp > num_points:
                        voxel_size_min = voxel_size
                    elif num_points_tmp < num_points:
                        voxel_size_max = voxel_size

                    voxel_size = (voxel_size_min + voxel_size_max) / 2

            partial_pcd = partial_pcd_tmp.select_by_index(np.random.choice(num_points_tmp, num_points, replace=False))

        partial_pcds += [partial_pcd]

    vis.destroy_window()

    # Check obtained partial point cloud with mesh
    if check_partial_pc:
        for partial_pcd in partial_pcds:
            o3d.visualization.draw_geometries([mesh, partial_pcd])

    # Convert open3d PointCloud to numpy array
    partial_pcs = np.stack([np.asarray(partial_pcd.points) for partial_pcd in partial_pcds])

    return partial_pcs


class PartialPointCloudExtractor:
    def __init__(self):
        # set offscreen rendering
        width = 128
        height = 128

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # Set intrinsic parameters
        fx = fy = 110.85125168
        cx = (width - 1) / 2
        cy = (height - 1) / 2

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    def extract(self, mesh, view_vecs, num_points):
        # set distance from object center to camera
        distance = np.linalg.norm(mesh.get_oriented_bounding_box().extent)

        # add mesh
        self.renderer.scene.add_geometry('mesh', mesh, o3d.visualization.rendering.MaterialRecord())

        # set camera poses
        view_unit_vecs = view_vecs / np.linalg.norm(view_vecs, axis=1, keepdims=True)

        cam_z_s = - view_unit_vecs

        while True:
            cam_x_s = -1 + 2 * np.random.rand(len(cam_z_s), 3)
            cam_x_s = cam_x_s - np.sum(cam_x_s*cam_z_s, axis=1, keepdims=True) * cam_z_s

            if np.linalg.norm(cam_x_s, axis=1).any() == 0:
                continue
            else:
                cam_x_s /= np.linalg.norm(cam_x_s, axis=1, keepdims=True)

                break

        cam_y_s = np.cross(cam_z_s, cam_x_s)
        cam_y_s /= np.linalg.norm(cam_y_s, axis=1, keepdims=True)

        cam_Ts = np.tile(np.eye(4), (len(view_vecs), 1, 1))
        cam_Ts[:, :3, :3] = np.stack([cam_x_s, cam_y_s, cam_z_s], axis=2)
        cam_Ts[:, :3, 3] = distance * view_unit_vecs

        # Get partial point clouds
        partial_pcds = []

        for cam_T in cam_Ts:
            # set extrinsic parameters
            extrinsic = np.linalg.inv(cam_T)

            # Set camera
            self.renderer.setup_camera(self.intrinsic, extrinsic)

            # Get depth image
            depth_image = self.renderer.render_to_depth_image(z_in_view_space=True)

            # get partial point cloud
            partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, self.intrinsic, extrinsic)

            pts = np.asarray(partial_pcd.points)
            pts = pts[~np.isnan(pts).any(1)]

            partial_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))

            # raise Exception if the number of points in point cloud is less than 'num_points'
            if len(np.asarray(partial_pcd.points)) < num_points:
                raise Exception("Point cloud has an insufficient number of points. Increase visualizer window width and height.")

            # downsample point cloud to match the number of points with 'num_points'
            else:
                voxel_size = 0.5
                voxel_size_min = 0
                voxel_size_max = 1

                while True:
                    partial_pcd_tmp = partial_pcd.voxel_down_sample(voxel_size)

                    num_points_tmp = len(np.asarray(partial_pcd_tmp.points))

                    if num_points_tmp - num_points >= 0 and num_points_tmp - num_points < 100:
                        break
                    else:
                        if num_points_tmp > num_points:
                            voxel_size_min = voxel_size
                        elif num_points_tmp < num_points:
                            voxel_size_max = voxel_size

                        voxel_size = (voxel_size_min + voxel_size_max) / 2

                partial_pcd = partial_pcd_tmp.select_by_index(np.random.choice(num_points_tmp, num_points, replace=False))

            partial_pcds += [partial_pcd]

        # convert open3d PointCloud to numpy array
        partial_pcs = np.stack([np.asarray(partial_pcd.points) for partial_pcd in partial_pcds])

        # Delete mesh
        self.renderer.scene.remove_geometry('mesh')

        return partial_pcs
