import torch
import time
import logging
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import os

from utils.average_meter import AverageMeter
from utils.mesh import generate_grasp_scene_list, meshes_to_numpy


NUM_GRASPS = 100


class GraspPoseGeneratorTrainer:
    def __init__(self, cfg, device, dataloaders, model, losses, optimizer, metrics, logger):
        self.cfg = cfg
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.device = device
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.logger = logger
        self.metrics = metrics

        # Get logdir
        self.logdir = self.logger.writer.file_writer.get_logdir()

        # Setup meters
        self.setup_meters()

        # Initialize performance dictionary
        self.setup_performance_dict()

    def setup_meters(self):
        # Setup time meter
        self.time_meter = AverageMeter()

        # Setup scalar meters for train
        for data in self.train_loader:
            break

        for key, val in data.items():
            if type(val) == torch.Tensor:
                data[key] = val.to(self.device)
            elif type(val) == list:
                data[key] = [v.to(self.device) for v in val]

        with torch.no_grad():
            results_train = self.model.step(data, self.losses, 'train')
            results_val = self.model.step(data, self.losses, 'val')

        self.train_meters = {key: AverageMeter() for key in results_train.keys() if 'scalar' in key}
        self.val_meters = {key: AverageMeter() for key in results_val.keys() if 'scalar' in key}

        # Setup metric meters
        self.metric_meters = {key: AverageMeter() for key in self.metrics.keys()}

    def setup_performance_dict(self):
        self.performances = {'val_loss': torch.inf}

        for criterion in self.cfg.criteria:
            assert criterion.name in self.metrics.keys(), f"Criterion {criterion.name} not in metrics keys {self.metrics.keys()}."

            if criterion.better == 'higher':
                self.performances[criterion.name] = 0
            elif criterion.better == 'lower':
                self.performances[criterion.name] = torch.inf
            else:
                raise ValueError(f"Criterion better with {criterion.better} value is not supported. Choose 'higher' or 'lower'.")

    def run(self):
        # Initialize
        iter = 0

        # Start learning
        for epoch in range(1, self.cfg.num_epochs+1):
            for data in self.train_loader:
                iter += 1

                # Training
                results_train = self.train(data)

                # Print
                if iter % self.cfg.print_interval == 0:
                    self.print(results_train, epoch, iter)

                # Validation
                if iter % self.cfg.val_interval == 0:
                    self.validate(epoch, iter)

                # Evaluation
                if iter % self.cfg.eval_interval == 0:
                    self.evaluate(epoch, iter)

                # Visualization
                if iter % self.cfg.vis_interval == 0:
                    self.visualize(epoch, iter)

                # Save
                if iter % self.cfg.save_interval == 0:
                    self.save(epoch, iter)

    def train(self, data):
        # Initialize
        self.model.train()
        self.optimizer.zero_grad()

        # Setup input
        for key, val in data.items():
            if type(val) == torch.Tensor:
                data[key] = val.to(self.device)
            elif type(val) == list:
                data[key] = [v.to(self.device) for v in val]

        # Step
        time_start = time.time()

        results = self.model.step(data, self.losses, 'train', self.optimizer)

        time_end = time.time()

        # Update time meter
        self.time_meter.update(time_end - time_start)

        # Update train meters
        for key, meter in self.train_meters.items():
            meter.update(results[key], n=len(data['pc']))

        return results

    def print(self, results, epoch, iter):
        # Get averaged train results
        for key, meter in self.train_meters.items():
            results[key] = meter.avg

        # Log averaged train results
        self.logger.log(results, iter)

        # Print averaged train results
        msg = f"[   Training  ] epoch: {epoch}/{self.cfg.num_epochs}, iter: {iter}, "
        msg += ", ".join([f"{key.split('/')[-1]}: {meter.avg:.4f}" for key, meter in self.train_meters.items()])
        msg += f", elapsed time: {self.time_meter.sum:.4f}"

        print(msg)
        logging.info(msg)

        # Reset time meter and train meters
        self.time_meter.reset()

        for key, meter in self.train_meters.items():
            meter.reset()

    def validate(self, epoch, iter):
        # Initialize
        self.model.eval()

        time_start = time.time()

        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validating ...", leave=False):
                # Setup input
                for key, val in data.items():
                    if type(val) == torch.Tensor:
                        data[key] = val.to(self.device)
                    elif type(val) == list:
                        data[key] = [v.to(self.device) for v in val]

                # Step
                results = self.model.step(data, self.losses, 'val')

                # Update validation meters
                for key, meter in self.val_meters.items():
                    meter.update(results[key], n=len(data['pc']))

        time_end = time.time()

        # Get averaged validation results
        for key, meter in self.val_meters.items():
            results[key] = meter.avg

        # Log averaged validation results
        self.logger.log(results, iter)

        # Print averaged validation results
        msg = f"[  Validation ] epoch: {epoch}/{self.cfg.num_epochs}, iter: {iter}, "
        msg += ", ".join([f"{key.split('/')[-1]}: {meter.avg:.4f}" for key, meter in self.val_meters.items()])
        msg += f", elapsed time: {time_end-time_start:.4f}"

        print(msg)
        logging.info(msg)

        # Determine best validation loss
        val_loss = self.val_meters['scalar/val/loss'].avg

        if val_loss < self.performances['val_loss']:
            # Save model
            self.save(epoch, iter, criterion='val_loss', data={'val_loss': val_loss})

            # Update best validation loss
            self.performances['val_loss'] = val_loss

        # Reset meters
        for key, meter in self.val_meters.items():
            meter.reset()

    def evaluate(self, epoch, iter):
        # Initialize
        self.model.eval()

        # Get dataset and scale
        pc_list_types = self.val_loader.dataset.pc_list_types
        Ts_grasp_list_types = self.val_loader.dataset.Ts_grasp_list_types
        mesh_list_types = deepcopy(self.val_loader.dataset.mesh_list_types)

        scale = self.val_loader.dataset.scale if hasattr(self.val_loader.dataset, 'scale') else 1

        time_start = time.time()

        # Iterate object types
        for pc_list_objs, Ts_grasp_list_objs, mesh_list_objs in zip(tqdm(pc_list_types, desc="Evaluating for object types ...", leave=False), Ts_grasp_list_types, mesh_list_types):
            # Setup metric meters for objects
            metric_meters_objs = {key: AverageMeter() for key in self.metrics.keys()}

            # Iterate objects
            for pc_rots, Ts_grasp_rots_target, mesh_list_rots in zip(tqdm(pc_list_objs, desc="Evaluating for objects ...", leave=False), Ts_grasp_list_objs, mesh_list_objs):
                # Setup metric meters for rotations
                metric_meters_rots = {key: AverageMeter() for key in self.metrics.keys()}

                # Setup input
                pc_rots = torch.Tensor(pc_rots).to(self.device)
                Ts_grasp_rots_target = torch.Tensor(Ts_grasp_rots_target).to(self.device)
                nums_grasps = torch.tensor([len(Ts_grasp_target) for Ts_grasp_target in Ts_grasp_rots_target], device=pc_rots.device)

                # Generate grasp poses
                Ts_grasp_rots_pred = self.model.sample(pc_rots, nums_grasps)

                # Rescale grasp poses and mesh
                for Ts_grasp_pred, Ts_grasp_target, mesh in zip(Ts_grasp_rots_pred, Ts_grasp_rots_target, mesh_list_rots):
                    Ts_grasp_pred[:, :3, 3] /= scale
                    Ts_grasp_target[:, :3, 3] /= scale
                    mesh.scale(1/scale, center=(0, 0, 0))

                # Compute metrics for rotations
                for Ts_grasp_pred, Ts_grasp_target, mesh in zip(Ts_grasp_rots_pred, Ts_grasp_rots_target, mesh_list_rots):
                    for key, metric in self.metrics.items():
                        if key == 'collision_rate':
                            # Get indices for sampling grasp poses for simulation
                            assert NUM_GRASPS <= len(Ts_grasp_pred), f"Number of grasps for simulation ({NUM_GRASPS}) must be less than or equal to the number of grasps predicted ({len(Ts_grasp_pred)})."

                            idxs = torch.randperm(len(Ts_grasp_pred))[:NUM_GRASPS]

                            metric_meters_rots[key].update(metric(mesh, Ts_grasp_pred[idxs]))
                        else:
                            metric_meters_rots[key].update(metric(Ts_grasp_pred, Ts_grasp_target))

                # Compute metrics for objects
                for key, meter in metric_meters_objs.items():
                    meter.update(metric_meters_rots[key].avg)

            # Compute metrics for object types
            for key, meter in self.metric_meters.items():
                meter.update(metric_meters_objs[key].avg)

        time_end = time.time()

        # Get averaged evaluation results
        results = {}

        for key, meter in self.metric_meters.items():
            results[f'scalar/metrics/{key}'] = meter.avg

        # Log averaged evaluation results
        self.logger.log(results, iter)

        # Print averaged evaluation results
        msg = f"[  Evaluation ] epoch: {epoch}/{self.cfg.num_epochs}, iter: {iter}, "
        msg += ", ".join([f"{key}: {meter.avg:.4f}" for key, meter in self.metric_meters.items()])
        msg += f", elapsed time: {time_end-time_start:.4f}"

        print(msg)
        logging.info(msg)

        # Save model if best evaluation performance
        for criterion in self.cfg.criteria:
            # Determine best performance
            performance = self.metric_meters[criterion.name].avg

            if criterion.better == 'higher' and performance > self.performances[criterion.name]:
                best = True
            elif criterion.better == 'lower' and performance < self.performances[criterion.name]:
                best = True
            else:
                best = False

            if best:
                # Save model
                self.save(epoch, iter, criterion=criterion.name, data={criterion.name: performance})

                # Update best validation loss
                self.performances[criterion.name] = performance

        # Reset metric meters
        for key, meter in self.metric_meters.items():
            meter.reset()

    def visualize(self, epoch, iter):
        # Initialize
        self.model.eval()

        time_start = time.time()

        mesh_list = []
        pc_list = []
        Ts_grasp_pred_list = []
        Ts_grasp_target_list = []

        # Get random data indices
        idxs = np.random.choice(self.val_loader.dataset.num_scenes, size=3, replace=False)

        # Get scale
        scale = self.val_loader.dataset.scale if hasattr(self.val_loader.dataset, 'scale') else 1

        for idx in idxs:
            idx_type, idx_obj, idx_rot = np.where(self.val_loader.dataset.scene_idxs==idx)

            idx_type = idx_type.item()
            idx_obj = idx_obj.item()
            idx_rot = idx_rot.item()

            # Get input
            mesh = deepcopy(self.val_loader.dataset.mesh_list_types[idx_type][idx_obj][idx_rot])
            pc = self.val_loader.dataset.pc_list_types[idx_type][idx_obj][idx_rot]
            Ts_grasp_target = self.val_loader.dataset.Ts_grasp_list_types[idx_type][idx_obj][idx_rot]

            # Sample ground-truth grasp poses
            idxs_grasp = np.random.choice(len(Ts_grasp_target), size=10, replace=False)
            Ts_grasp_target = Ts_grasp_target[idxs_grasp]

            # Append data to list
            mesh_list += [mesh]
            pc_list += [torch.Tensor(pc)]
            Ts_grasp_target_list += [Ts_grasp_target]

        # Setup input
        pc = torch.stack(pc_list).to(self.device)
        nums_grasps = torch.tensor([10, 10, 10], device=self.device)

        # Generate grasp poses
        Ts_grasp_pred_list = self.model.sample(pc, nums_grasps)
        Ts_grasp_pred_list = [Ts_grasp_pred.cpu().numpy() for Ts_grasp_pred in Ts_grasp_pred_list]

        # Rescale mesh and grasp poses
        for mesh, Ts_grasp_pred, Ts_grasp_target in zip(mesh_list, Ts_grasp_pred_list, Ts_grasp_target_list):
            mesh.scale(1/scale, center=(0, 0, 0))
            Ts_grasp_pred[:, :3, 3] /= scale
            Ts_grasp_target[:, :3, 3] /= scale

        # Generate scene
        scene_list_pred = generate_grasp_scene_list(mesh_list, Ts_grasp_pred_list)
        scene_list_target = generate_grasp_scene_list(mesh_list, Ts_grasp_target_list)

        # Get vertices, triangles and colors
        vertices_pred, triangles_pred, colors_pred = meshes_to_numpy(scene_list_pred)
        vertices_target, triangles_target, colors_target = meshes_to_numpy(scene_list_target)

        time_end = time.time()

        # Get visualization results
        results = {
            'mesh/pred': {'vertices': vertices_pred, 'colors': colors_pred, 'faces': triangles_pred},
            'mesh/target': {'vertices': vertices_target, 'colors': colors_target, 'faces': triangles_target}
        }

        # Log visualization results
        self.logger.log(results, iter)

        # Print visualization status
        msg = f"[Visualization] epoch: {epoch}/{self.cfg.num_epochs}, iter: {iter}"
        msg += f", elapsed time: {time_end-time_start:.4f}"

        print(msg)
        logging.info(msg)

    def save(self, epoch, iter, criterion=None, data={}):
        # Set save name
        if criterion is None:
            save_name = f'model_iter_{iter}.pkl'
        else:
            save_name = f'model_best_{criterion}.pkl'

        # Construct object to save
        object = {
            'epoch': epoch,
            'iter': iter,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        object.update(data)

        # Save object
        save_path = os.path.join(self.logdir, save_name)

        torch.save(object, save_path)

        # Print save status
        string = f"[     Save    ] epoch: {epoch}/{self.cfg.num_epochs}, iter: {iter}, save {save_name}"

        if criterion is not None:
            string += f", {criterion}: {data[criterion]:.6f} / best_{criterion}: {self.performances[criterion]:.6f}"

        print(string)
        logging.info(string)


class PartialGraspPoseGeneratorTrainer(GraspPoseGeneratorTrainer):
    def evaluate(self, epoch, iter):
        # Initialize
        self.model.eval()

        # Get dataset and scale
        partial_pc_list_types = self.val_loader.dataset.partial_pc_list_types
        Ts_grasp_list_types = self.val_loader.dataset.Ts_grasp_list_types
        mesh_list_types = deepcopy(self.val_loader.dataset.mesh_list_types)

        scale = self.val_loader.dataset.scale if hasattr(self.val_loader.dataset, 'scale') else 1

        time_start = time.time()

        # Iterate object types
        for partial_pc_list_objs, Ts_grasp_list_objs, mesh_list_objs in zip(tqdm(partial_pc_list_types, desc="Evaluating for object types ...", leave=False), Ts_grasp_list_types, mesh_list_types):
            # Setup metric meters for objects
            metric_meters_objs = {key: AverageMeter() for key in self.metrics.keys()}

            # Iterate objects
            for partial_pc_rots, Ts_grasp_rots_target, mesh_list_rots in zip(tqdm(partial_pc_list_objs, desc="Evaluating for objects ...", leave=False), Ts_grasp_list_objs, mesh_list_objs):
                # Setup metric meters for rotations
                metric_meters_rots = {key: AverageMeter() for key in self.metrics.keys()}

                # Setup input
                Ts_grasp_rots_target = torch.Tensor(Ts_grasp_rots_target).to(self.device)

                # Iterate rotations
                for partial_pc_views, Ts_grasp_views_target, mesh_list_views in zip(tqdm(partial_pc_rots, desc="Evaluating for rotations ...", leave=False), Ts_grasp_rots_target, mesh_list_rots):
                    # Setup metric meters for viewpoints
                    metric_meters_views = {key: AverageMeter() for key in self.metrics.keys()}

                    # Setup input
                    partial_pc_views = torch.Tensor(partial_pc_views).to(self.device)
                    nums_grasps = torch.tensor([Ts_grasp_views_target.shape[1]]*len(partial_pc_views), device=partial_pc_views.device)

                    # Generate grasp poses
                    Ts_grasp_views_pred = self.model.sample(partial_pc_views, nums_grasps)

                    for Ts_grasp_pred, Ts_grasp_target, mesh in zip(Ts_grasp_views_pred, Ts_grasp_views_target, mesh_list_views):
                        # Rescale grasp poses and mesh
                        Ts_grasp_pred[:, :3, 3] /= scale
                        Ts_grasp_target[:, :3, 3] /= scale
                        mesh.scale(1/scale, center=(0, 0, 0))

                        # Compute metrics for viewpoints
                        for key, metric in self.metrics.items():
                            if key == 'collision_rate':
                                # Get indices for sampling grasp poses for simulation
                                assert NUM_GRASPS <= len(Ts_grasp_pred), f"Number of grasps for simulation ({NUM_GRASPS}) must be less than or equal to the number of grasps predicted ({len(Ts_grasp_pred)})."

                                idxs = torch.randperm(len(Ts_grasp_pred))[:NUM_GRASPS]

                                metric_meters_views[key].update(metric(mesh, Ts_grasp_pred[idxs]))
                            else:
                                metric_meters_views[key].update(metric(Ts_grasp_pred, Ts_grasp_target))

                    # Compute metrics for rotations
                    for key, meter in metric_meters_objs.items():
                        meter.update(metric_meters_views[key].avg)

                # Compute metrics for objects
                for key, meter in metric_meters_objs.items():
                    meter.update(metric_meters_rots[key].avg)

            # Compute metrics for object types
            for key, meter in self.metric_meters.items():
                meter.update(metric_meters_objs[key].avg)

        time_end = time.time()

        # Get averaged evaluation results
        results = {}

        for key, meter in self.metric_meters.items():
            results[f'scalar/metrics/{key}'] = meter.avg

        # Log averaged evaluation results
        self.logger.log(results, iter)

        # Print averaged evaluation results
        msg = f"[  Evaluation ] epoch: {epoch}/{self.cfg.num_epochs}, iter: {iter}, "
        msg += ", ".join([f"{key}: {meter.avg:.4f}" for key, meter in self.metric_meters.items()])
        msg += f", elapsed time: {time_end-time_start:.4f}"

        print(msg)
        logging.info(msg)

        # Save model if best evaluation performance
        for criterion in self.cfg.criteria:
            # Determine best performance
            performance = self.metric_meters[criterion.name].avg

            if criterion.better == 'higher' and performance > self.performances[criterion.name]:
                best = True
            elif criterion.better == 'lower' and performance < self.performances[criterion.name]:
                best = True
            else:
                best = False

            if best:
                # Save model
                self.save(epoch, iter, criterion=criterion.name, data={criterion.name: performance})

                # Update best validation loss
                self.performances[criterion.name] = performance

        # Reset metric meters
        for key, meter in self.metric_meters.items():
            meter.reset()

    def visualize(self, epoch, iter):
        # Initialize
        self.model.eval()

        time_start = time.time()

        mesh_list = []
        partial_pc_list = []
        Ts_grasp_pred_list = []
        Ts_grasp_target_list = []

        # Get random data indices
        idxs = np.random.choice(self.val_loader.dataset.num_scenes, size=3, replace=False)

        # Get scale
        scale = self.val_loader.dataset.scale if hasattr(self.val_loader.dataset, 'scale') else 1

        for idx in idxs:
            idx_type, idx_obj, idx_rot, idx_view = np.where(self.val_loader.dataset.scene_idxs==idx)

            idx_type = idx_type.item()
            idx_obj = idx_obj.item()
            idx_rot = idx_rot.item()
            idx_view = idx_view.item()

            # Get input
            mesh = deepcopy(self.val_loader.dataset.mesh_list_types[idx_type][idx_obj][idx_rot][idx_view])
            partial_pc = self.val_loader.dataset.partial_pc_list_types[idx_type][idx_obj][idx_rot][idx_view]
            Ts_grasp_target = self.val_loader.dataset.Ts_grasp_list_types[idx_type][idx_obj][idx_rot][idx_view]

            # Sample ground-truth grasp poses
            idxs_grasp = np.random.choice(len(Ts_grasp_target), size=10, replace=False)
            Ts_grasp_target = Ts_grasp_target[idxs_grasp]

            # Append data to list
            mesh_list += [mesh]
            partial_pc_list += [torch.Tensor(partial_pc)]
            Ts_grasp_target_list += [Ts_grasp_target]

        # Setup input
        partial_pc = torch.stack(partial_pc_list).to(self.device)
        nums_grasps = torch.tensor([10, 10, 10], device=self.device)

        # Generate grasp poses
        Ts_grasp_pred_list = self.model.sample(partial_pc, nums_grasps)
        Ts_grasp_pred_list = [Ts_grasp_pred.cpu().numpy() for Ts_grasp_pred in Ts_grasp_pred_list]

        # Rescale mesh and grasp poses
        for mesh, Ts_grasp_pred, Ts_grasp_target in zip(mesh_list, Ts_grasp_pred_list, Ts_grasp_target_list):
            mesh.scale(1/scale, center=(0, 0, 0))
            Ts_grasp_pred[:, :3, 3] /= scale
            Ts_grasp_target[:, :3, 3] /= scale

        # Generate scene
        scene_list_pred = generate_grasp_scene_list(mesh_list, Ts_grasp_pred_list)
        scene_list_target = generate_grasp_scene_list(mesh_list, Ts_grasp_target_list)

        # Get vertices, triangles and colors
        vertices_pred, triangles_pred, colors_pred = meshes_to_numpy(scene_list_pred)
        vertices_target, triangles_target, colors_target = meshes_to_numpy(scene_list_target)

        time_end = time.time()

        # Get visualization results
        results = {
            'mesh/pred': {'vertices': vertices_pred, 'colors': colors_pred, 'faces': triangles_pred},
            'mesh/target': {'vertices': vertices_target, 'colors': colors_target, 'faces': triangles_target}
        }

        # Log visualization results
        self.logger.log(results, iter)

        # Print visualization status
        msg = f"[Visualization] epoch: {epoch}/{self.cfg.num_epochs}, iter: {iter}"
        msg += f", elapsed time: {time_end-time_start:.4f}"

        print(msg)
        logging.info(msg)
