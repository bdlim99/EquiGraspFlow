from copy import deepcopy
import open3d as o3d
import numpy as np


def generate_grasp_scene_list(mesh_list, Ts_grasp_list):
    scene_list = []

    for mesh, Ts_grasp in zip(mesh_list, Ts_grasp_list):
        scene = deepcopy(mesh)

        for T_grasp in Ts_grasp:
            mesh_base_1 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.066, resolution=6, split=1)
            T_base_1 = np.eye(4)
            T_base_1[:3, 3] = [0, 0, 0.033]
            mesh_base_1.transform(T_base_1)

            mesh_base_2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.082, resolution=6, split=1)
            T_base_2 = np.eye(4)
            T_base_2[:3, :3] = mesh_base_2.get_rotation_matrix_from_xyz([0, np.pi/2, 0])
            T_base_2[:3, 3] = [0, 0, 0.066]
            mesh_base_2.transform(T_base_2)

            mesh_left_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.046, resolution=6, split=1)
            T_left_finger = np.eye(4)
            T_left_finger[:3, 3] = [-0.041, 0, 0.089]
            mesh_left_finger.transform(T_left_finger)

            mesh_right_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.046, resolution=6, split=1)
            T_right_finger = np.eye(4)
            T_right_finger[:3, 3] = [0.041, 0, 0.089]
            mesh_right_finger.transform(T_right_finger)

            mesh_gripper = mesh_base_1 + mesh_base_2 + mesh_left_finger + mesh_right_finger
            mesh_gripper.transform(T_grasp)

            scene += mesh_gripper

        scene.compute_vertex_normals()
        scene.paint_uniform_color([0.5, 0.5, 0.5])

        scene_list += [scene]

    return scene_list


def meshes_to_numpy(scenes):
    # Initialize
    vertices_np = []
    triangles_np = []
    colors_np = []

    # Get maximum number of vertices and triangles
    max_num_vertices = max([len(scene.vertices) for scene in scenes])
    max_num_triangles = max([len(scene.triangles) for scene in scenes])

    # Match dimension between batches for Tensorboard
    for scene in scenes:
        diff_num_vertices = max_num_vertices - len(scene.vertices)
        diff_num_triangles = max_num_triangles - len(scene.triangles)

        vertices_np += [np.concatenate((np.asarray(scene.vertices), np.zeros((diff_num_vertices, 3))), axis=0)]
        triangles_np += [np.concatenate((np.asarray(scene.triangles), np.zeros((diff_num_triangles, 3))), axis=0)]
        colors_np += [np.concatenate((255 * np.asarray(scene.vertex_colors), np.zeros((diff_num_vertices, 3))), axis=0)]

    # Stack to single numpy array
    vertices_np = np.stack(vertices_np, axis=0)
    triangles_np = np.stack(triangles_np, axis=0)
    colors_np = np.stack(colors_np, axis=0)

    return vertices_np, triangles_np, colors_np
