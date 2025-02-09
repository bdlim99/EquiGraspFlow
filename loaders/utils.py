import os
import open3d as o3d
import pickle
import numpy as np


DATASET_DIR = 'dataset'


def load_grasp_poses(data):
    grasps = data['grasps/transforms'][()]
    success = data['grasps/qualities/flex/object_in_gripper'][()]

    grasps_good = grasps[success==1]

    return grasps_good


def load_mesh(data):
    mesh_path = data['object/file'][()].decode('utf-8')
    mesh_scale = data['object/scale'][()]

    mesh = o3d.io.read_triangle_mesh(os.path.join(DATASET_DIR, mesh_path))

    mesh.scale(mesh_scale, center=(0, 0, 0))

    return mesh
