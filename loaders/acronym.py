import torch
import numpy as np
from scipy.spatial.transform import Rotation
import os
from tqdm import tqdm
import h5py
from copy import deepcopy

from loaders.utils import load_grasp_poses, load_mesh
from utils.Lie import super_fibonacci_spiral, get_fibonacci_sphere
from utils.partial_point_cloud import get_partial_point_clouds


DATASET_DIR = 'dataset'
NUM_GRASPS = 100


class AcronymFullPointCloud(torch.utils.data.Dataset):
    def __init__(self, split, obj_types, augmentation, scale, num_pts=1024, num_rots=1):
        # Initialize
        self.len_dataset = 0
        self.split = split
        self.num_pts = num_pts
        self.augmentation = augmentation
        self.scale = scale
        self.num_rots = num_rots
        self.obj_types = obj_types

        # Initialize maximum number of objects
        self.max_num_objs = 0

        # Get evenly distributed rotations for validation and test splits
        if split in ['val', 'test']:
            if augmentation == 'None':
                self.Rs = np.expand_dims(np.eye(3), axis=0)
            elif augmentation == 'z':
                degree = (np.arange(num_rots) / num_rots) * (2 * np.pi)
                self.Rs = Rotation.from_rotvec(degree * np.array([0, 0, 1])).as_matrix()
            elif augmentation == 'SO3':
                self.Rs = super_fibonacci_spiral(num_rots)
            else:
                raise ValueError("Choose augmentation from ['None', 'z', 'SO3'].")
        else:
            assert num_rots == 1, "Number of rotations must be 1 in train set."

            self.Rs = np.expand_dims(np.eye(3), axis=0)

        Ts = np.tile(np.eye(4), (num_rots, 1, 1))
        Ts[:, :3, :3] = self.Rs

        # Initialize data indices
        data_idxs_types = []

        # Initialize lists
        self.mesh_list_types = []
        self.Ts_grasp_list_types = []

        self.pc_list_types = []

        self.obj_idxs_types = []

        for obj_type in tqdm(obj_types, desc="Iterating object types ...", leave=False):
            # Get data filenames
            filenames = sorted(os.listdir(os.path.join(DATASET_DIR, 'grasps', obj_type)))

            # Get object indices for the split
            obj_idxs = np.load(os.path.join(DATASET_DIR, 'splits', obj_type, f'idxs_{split}.npy'))

            # Initialize data indices
            data_idxs_objs = []

            # Initialize lists
            mesh_list_objs = []
            Ts_grasp_list_objs = []

            pc_list_objs = []

            obj_idxs_objs = []

            for obj_idx in tqdm(obj_idxs, desc="Iterating objects ...", leave=False):
                # Get data filename
                filename = filenames[obj_idx]

                # Load data
                data = h5py.File(os.path.join(DATASET_DIR, 'grasps', obj_type, filename))

                # Load grasp poses
                Ts_grasp = load_grasp_poses(data)

                # Continue if grasp poses are not enough
                if len(Ts_grasp) < NUM_GRASPS:
                    continue
                else:
                    obj_idxs_objs += [obj_idx]

                # Load mesh
                mesh = load_mesh(data)

                # Scale
                mesh.scale(scale, center=(0, 0, 0))
                Ts_grasp[:, :3, 3] *= scale

                # Sample point cloud
                pc = np.asarray(mesh.sample_points_uniformly(num_pts).points).T

                # Translate to the center of the point cloud
                center = pc.mean(axis=1)
                mesh.translate(-center)
                pc -= np.expand_dims(center, axis=1)
                Ts_grasp[:, :3, 3] -= center

                # Rotate mesh
                mesh_list_rots = []

                for R in tqdm(self.Rs, desc="Iterating rotations ...", leave=False):
                    mesh_rot = deepcopy(mesh)
                    mesh_rot.rotate(R, center=(0, 0, 0))

                    mesh_list_rots += [mesh_rot]

                # Rotate the other data
                pc_rots = self.Rs @ pc
                Ts_grasp_rots = np.einsum('rij,njk->rnik', Ts, Ts_grasp)

                # Fill data indices
                data_idxs_objs += [list(range(self.len_dataset, self.len_dataset+num_rots))]

                # Append data
                mesh_list_objs += [mesh_list_rots]
                Ts_grasp_list_objs += [Ts_grasp_rots]

                pc_list_objs += [pc_rots]

                # Increase number of data
                self.len_dataset += num_rots

            # Append data
            data_idxs_types += [data_idxs_objs]

            self.mesh_list_types += [mesh_list_objs]
            self.Ts_grasp_list_types += [Ts_grasp_list_objs]

            self.pc_list_types += [pc_list_objs]

            self.obj_idxs_types += [obj_idxs_objs]

            # Update maximum number of objects
            if len(obj_idxs_objs) > self.max_num_objs:
                self.max_num_objs = len(obj_idxs_objs)

        # Convert data indices from lists to numpy array
        self.data_idxs = np.full((len(obj_types), self.max_num_objs, num_rots), -1)

        for i, data_idxs_objs in enumerate(data_idxs_types):
            self.data_idxs[i, :len(data_idxs_objs)] = data_idxs_objs

        # Setup scene indices
        self.scene_idxs = self.data_idxs
        self.num_scenes = self.scene_idxs.max() + 1

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        # Get type, object, and rotation indices
        idx_type, idx_obj, idx_rot = np.where(self.data_idxs==idx)

        idx_type = idx_type.item()
        idx_obj = idx_obj.item()
        idx_rot = idx_rot.item()

        # Load grasp poses
        Ts_grasp = self.Ts_grasp_list_types[idx_type][idx_obj][idx_rot].copy()

        if self.split == 'train':
            # Load mesh
            mesh = self.mesh_list_types[idx_type][idx_obj][idx_rot]

            # Sample point cloud
            pc = np.asarray(mesh.sample_points_uniformly(self.num_pts).points).T

            # Translate to the point cloud center
            center = pc.mean(axis=1)
            pc -= np.expand_dims(center, axis=1)
            Ts_grasp[:, :3, 3] -= center

            # Rotate data
            if self.augmentation != 'None':
                if self.augmentation == 'z':
                    # Randomly rotate around z-axis
                    degree = np.random.rand() * 2 * np.pi
                    R = Rotation.from_rotvec(degree * np.array([0, 0, 1])).as_matrix()
                elif self.augmentation == 'SO3':
                    # Randomly rotate
                    R = Rotation.random().as_matrix()
                else:
                    raise ValueError("Choose augmentation from ['None', 'z', 'SO3'].")

                T = np.eye(4)
                T[:3, :3] = R

                pc = R @ pc
                Ts_grasp = T @ Ts_grasp
        else:
            # Load point cloud
            pc = self.pc_list_types[idx_type][idx_obj][idx_rot]

        return {'pc': torch.Tensor(pc), 'Ts_grasp': torch.Tensor(Ts_grasp)}


class AcronymPartialPointCloud(torch.utils.data.Dataset):
    def __init__(self, split, obj_types, augmentation, scale, num_pts=512, num_rots=1, num_views=1):
        # Initialize
        self.len_dataset = 0
        self.split = split
        self.num_pts = num_pts
        self.augmentation = augmentation
        self.scale = scale
        self.num_rots = num_rots
        self.num_views = num_views
        self.obj_types = obj_types

        # Initialize maximum number of objects
        self.max_num_objs = 0

        # Get rotations
        if split in ['val', 'test']:
            if augmentation == 'None':
                self.Rs = np.expand_dims(np.eye(3), axis=0)
            elif augmentation == 'z':
                degree = (np.arange(num_rots) / num_rots) * (2 * np.pi)
                self.Rs = Rotation.from_rotvec(degree * np.array([0, 0, 1])).as_matrix()
            elif augmentation == 'SO3':
                self.Rs = super_fibonacci_spiral(num_rots)
            else:
                assert augmentation == 'None', "Choose augmentation from ['None', 'z', 'SO3']."
        else:
            assert num_rots == 1, "Number of rotations must be 1 in train set."

            self.Rs = np.expand_dims(np.eye(3), axis=0)

        Ts = np.tile(np.eye(4), (num_rots, 1, 1))
        Ts[:, :3, :3] = self.Rs

        # Get viewpoint vector
        if split in ['val', 'test']:
            view_vecs = get_fibonacci_sphere(num_views)
        else:
            assert num_views == 1, "Number of viewpoint vector must be 1 in train set."

            view_vecs = np.array([[0, 0, 1]])

        # Initialize data indices
        data_idxs_types = []

        # Initialize lists
        self.mesh_list_types = []
        self.Ts_grasp_list_types = []

        self.partial_pc_list_types = []

        self.obj_idxs_types = []

        for obj_type in tqdm(obj_types, desc="Iterating object types ...", leave=False):
            # Get data filenames
            filenames = sorted(os.listdir(os.path.join(DATASET_DIR, 'grasps', obj_type)))

            # Get object indices for the split
            obj_idxs = np.load(os.path.join(DATASET_DIR, 'splits', obj_type, f'idxs_{split}.npy'))

            # Initialize data indices
            data_idxs_objs = []

            # Initialize lists
            mesh_list_objs = []
            Ts_grasp_list_objs = []

            partial_pc_list_objs = []

            obj_idxs_objs = []

            for obj_idx in tqdm(obj_idxs, desc="Iterating objects ...", leave=False):
                # Get data filename
                filename = filenames[obj_idx]

                # Load data
                data = h5py.File(os.path.join(DATASET_DIR, 'grasps', obj_type, filename))

                # Load grasp poses
                Ts_grasp = load_grasp_poses(data)

                # Continue if grasp poses are not enough
                if len(Ts_grasp) < NUM_GRASPS:
                    continue
                else:
                    obj_idxs_objs += [obj_idx]

                # Load mesh
                mesh = load_mesh(data)

                # Translate to the center of the mesh
                center = mesh.get_center()
                mesh.translate(-center)
                Ts_grasp[:, :3, 3] -= center

                # Scale
                mesh.scale(scale, center=(0, 0, 0))
                Ts_grasp[:, :3, 3] *= scale

                # Initialize data indices
                data_idxs_rots = []

                # Initialize lists
                mesh_list_rots = []
                partial_pc_list_rots = []

                for R in tqdm(self.Rs, desc="Iterating rotations ...", leave=False):
                    # Rotate mesh
                    mesh_rot = deepcopy(mesh)
                    mesh_rot.rotate(R, center=(0, 0, 0))

                    # Sample partial point clouds
                    partial_pc_views = get_partial_point_clouds(mesh_rot, view_vecs, num_pts, use_tqdm=True).transpose(0, 2, 1)

                    # Initialize mesh list
                    mesh_list_views = []

                    for partial_pc in partial_pc_views:
                        # Translate mesh to the center of the partial point cloud
                        mesh_view = deepcopy(mesh_rot)
                        mesh_view.translate(-partial_pc.mean(axis=1))

                        # Append mesh
                        mesh_list_views += [mesh_view]

                    # Fill data indices
                    data_idxs_rots += [list(range(self.len_dataset, self.len_dataset+num_views))]

                    # Append data
                    mesh_list_rots += [mesh_list_views]
                    partial_pc_list_rots += [partial_pc_views]

                    # Increase number of data
                    self.len_dataset += num_views

                # Stack partial point clouds
                partial_pc_rots = np.stack(partial_pc_list_rots)

                # Rotate grasp poses
                Ts_grasp_rots = np.einsum('rij,njk->rnik', Ts, Ts_grasp)

                # Translate to the center of the partial point clouds
                center_rots = partial_pc_rots.mean(axis=3)

                Ts_grasp_rots = np.expand_dims(Ts_grasp_rots, axis=1).repeat(num_views, axis=1)

                partial_pc_rots -= np.expand_dims(center_rots, axis=3)
                Ts_grasp_rots[:, :, :, :3, 3] -= np.expand_dims(center_rots, axis=2)

                # Append data
                data_idxs_objs += [data_idxs_rots]

                mesh_list_objs += [mesh_list_rots]
                Ts_grasp_list_objs += [Ts_grasp_rots]

                partial_pc_list_objs += [partial_pc_rots]

            # Append data
            data_idxs_types += [data_idxs_objs]

            self.mesh_list_types += [mesh_list_objs]
            self.Ts_grasp_list_types += [Ts_grasp_list_objs]

            self.partial_pc_list_types += [partial_pc_list_objs]

            self.obj_idxs_types += [obj_idxs_objs]

            # Update maximum number of objects
            if len(obj_idxs_objs) > self.max_num_objs:
                self.max_num_objs = len(obj_idxs_objs)

        # Convert data indices from lists to numpy array
        self.data_idxs = np.full((len(obj_types), self.max_num_objs, num_rots, num_views), -1)

        for i, data_idxs_objs in enumerate(data_idxs_types):
            self.data_idxs[i, :len(data_idxs_objs)] = data_idxs_objs

        # Setup scene indices
        self.scene_idxs = self.data_idxs
        self.num_scenes = self.scene_idxs.max() + 1

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        # Get type, object, and rotation indices
        idx_type, idx_obj, idx_rot, idx_view = np.where(self.data_idxs==idx)

        idx_type = idx_type.item()
        idx_obj = idx_obj.item()
        idx_rot = idx_rot.item()
        idx_view = idx_view.item()

        # Load grasp poses
        Ts_grasp = self.Ts_grasp_list_types[idx_type][idx_obj][idx_rot][idx_view].copy()

        if self.split == 'train':
            # Load mesh
            mesh = deepcopy(self.mesh_list_types[idx_type][idx_obj][idx_rot][idx_view])

            # Get random rotation
            if self.augmentation == 'None':
                R = np.eye(3)
            elif self.augmentation == 'z':
                degree = np.random.rand() * 2 * np.pi
                R = Rotation.from_rotvec(degree * np.array([0, 0, 1])).as_matrix()
            elif self.augmentation == 'SO3':
                R = Rotation.random().as_matrix()
            else:
                raise ValueError("Choose augmentation from ['None', 'z', 'SO3'].")

            T = np.eye(4)
            T[:3, :3] = R

            # Rotate mesh
            mesh.rotate(R, center=(0, 0, 0))

            # Sample partial point cloud
            while True:
                try:
                    view_vecs = -1 + 2 * np.random.rand(1, 3)
                    view_vecs = view_vecs / np.linalg.norm(view_vecs)

                    partial_pc = get_partial_point_clouds(mesh, view_vecs, self.num_pts)[0].T

                    break
                except:
                    pass

            # Rotate grasp poses
            Ts_grasp = T @ Ts_grasp

            # Translate to the center of the partial point cloud
            center = partial_pc.mean(axis=1)
            partial_pc -= np.expand_dims(center, axis=1)
            Ts_grasp[:, :3, 3] -= center
        else:
            # Load point cloud
            partial_pc = self.partial_pc_list_types[idx_type][idx_obj][idx_rot][idx_view]

        return {'pc': torch.Tensor(partial_pc), 'Ts_grasp': torch.Tensor(Ts_grasp)}
