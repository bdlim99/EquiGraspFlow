import argparse
from datetime import datetime
import os
from omegaconf import OmegaConf
import logging
import yaml
import random
import numpy as np
import torch
import plotly.graph_objects as go

from loaders import get_dataloader
from models import get_model
from metrics import get_metrics
from utils.visualization import PlotlySubplotsVisualizer


NUM_GRASPS = 25


def main(args, cfg):
    seed = cfg.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    # Setup testloader
    test_loader = get_dataloader('test', cfg.data.test)

    # Setup model
    model = get_model(cfg.model).to(cfg.device)

    # Setup metrics
    metrics = get_metrics(cfg.metrics)

    # Setup plotly visualizer
    visualizer = PlotlySubplotsVisualizer(rows=test_loader.dataset.num_rots, cols=test_loader.dataset.num_views)
    visualizer.fig.update_layout(height=2700)

    # Start test
    results = test(args, model, test_loader, metrics, cfg.device, visualizer)

    # Print results
    print_results(test_loader, results)

    # Save plotly figure
    save_figure(args.logdir, visualizer)


def test(args, model, test_loader, metrics, device, visualizer):
    # Initialize
    model.eval()

    # Get arguments
    logdir = args.logdir

    # Get dataset
    obj_types = test_loader.dataset.obj_types
    obj_idxs_types = test_loader.dataset.obj_idxs_types
    partial_pc_list_types = test_loader.dataset.partial_pc_list_types
    mesh_list_types = test_loader.dataset.mesh_list_types
    Ts_grasp_list_types = test_loader.dataset.Ts_grasp_list_types

    # Get scale, maximum number of objects and number of rotations
    scale = test_loader.dataset.scale if hasattr(test_loader.dataset, 'scale') else 1
    max_num_objs = test_loader.dataset.max_num_objs
    num_rots = test_loader.dataset.num_rots
    num_views = test_loader.dataset.num_views

    # Setup metric result arrays
    results = {key: np.full((len(partial_pc_list_types), max_num_objs, num_rots, num_views), np.nan) for key in list(metrics.keys())}

    # Setup labels for button in plotly figure
    visualizer.labels = []

    # Iterate
    for i, (obj_type, obj_idxs_objs, partial_pc_list_objs, Ts_grasp_list_objs, mesh_list_objs) in enumerate(zip(obj_types, obj_idxs_types, partial_pc_list_types, Ts_grasp_list_types, mesh_list_types)):
        for j, (obj_idx, partial_pc_rots, Ts_grasp_rots_target, mesh_list_rots) in enumerate(zip(obj_idxs_objs, partial_pc_list_objs, Ts_grasp_list_objs, mesh_list_objs)):
            # Setup input
            Ts_grasp_rots_target = torch.Tensor(Ts_grasp_rots_target).to(device)

            for k, (partial_pc_views, Ts_grasp_views_target, mesh_list_views) in enumerate(zip(partial_pc_rots, Ts_grasp_rots_target, mesh_list_rots)):
                # Setup input
                partial_pc_views = torch.Tensor(partial_pc_views).to(device)
                nums_grasps = torch.tensor([Ts_grasp_views_target.shape[1]]*len(partial_pc_views), device=partial_pc_views.device)

                # Sample grasp poses
                Ts_grasp_views_pred = model.sample(partial_pc_views, nums_grasps)

                # Compute metrics
                for l, (partial_pc, Ts_grasp_pred, Ts_grasp_target, mesh) in enumerate(zip(partial_pc_views, Ts_grasp_views_pred, Ts_grasp_views_target, mesh_list_views)):
                    # Setup message
                    msg = f"object type: {obj_type}, object index: {obj_idx}, rotation index: {k}, viewpoint index: {l}, "

                    # Rescale point cloud and grasp poses
                    partial_pc /= scale
                    Ts_grasp_pred[:, :3, 3] /= scale
                    Ts_grasp_target[:, :3, 3] /= scale
                    mesh.scale(1/scale, center=(0, 0, 0))

                    for key, metric in metrics.items():
                        # Compute metrics
                        result = metric(Ts_grasp_pred, Ts_grasp_target)

                        # Add result to message
                        msg += f"{key}: {result:.4f}, "

                        # Fill array
                        results[key][i, j, k, l] = result

                    # Print result message
                    print(msg)
                    logging.info(msg)

                    # Get indices for sampling grasp poses for simulation
                    idxs = torch.randperm(len(Ts_grasp_pred))[:NUM_GRASPS]

                    # Add mesh, partial point cloud, and gripper to visualizer
                    visualizer.add_mesh(mesh, row=k+1, col=l+1)

                    visualizer.add_pc(partial_pc.cpu().numpy().T, row=k+1, col=l+1)
                    visualizer.add_grippers(Ts_grasp_pred[idxs], color='grey', row=k+1, col=l+1)

            visualizer.labels += [f'{obj_type}_{obj_idx}']

    return results


def print_results(test_loader, results):
    # Get object types and object ids
    obj_types = test_loader.dataset.obj_types

    # Print results
    for idx_type, obj_type in enumerate(obj_types):
        msg = f"object type: {obj_type}"

        for key in results.keys():
            msg += f", {key}: {np.nanmean(results[key][idx_type]):.4f}"

        print(msg)
        logging.info(msg)


def save_figure(logdir, visualizer):
    # Get number of traces and number of subplots
    num_traces = len(visualizer.fig.data)
    num_subplots = visualizer.num_subplots

    # Make only the first scene visible
    for idx_trace in range(num_subplots*(2+NUM_GRASPS), num_traces):
        visualizer.fig.update_traces(visible=False, selector=idx_trace)

    # Make buttons list
    buttons = []

    for idx_scene, label in enumerate(visualizer.labels):
        # Initialize visibility list
        visibility = num_traces * [False]

        # Make only the selected scene visible
        for idx_trace in range(num_subplots*(2+NUM_GRASPS)*idx_scene, num_subplots*(2+NUM_GRASPS)*(idx_scene+1)):
            visibility[idx_trace] = True

        # Make and append button
        button = dict(label=label, method='restyle', args=[{'visible': visibility}])

        buttons += [button]

    # Update buttons
    visualizer.fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=buttons)])

    # Save figure
    visualizer.fig.write_json(os.path.join(logdir, 'visualizations.json'))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_result_path', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='test_results')
    parser.add_argument('--run', type=str, default=datetime.now().strftime('%Y%m%d-%H%M'))

    args = parser.parse_args()

    # Load config
    config_filename = [file for file in os.listdir(args.train_result_path) if file.endswith('.yml')][0]

    cfg = OmegaConf.load(os.path.join(args.train_result_path, config_filename))

    # Setup checkpoint
    cfg.model.checkpoint = os.path.join(args.train_result_path, args.checkpoint)

    # Setup device
    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'

    # Setup logdir
    config_basename = os.path.splitext(config_filename)[0]

    args.logdir = os.path.join(args.logdir, config_basename, args.run)

    os.makedirs(args.logdir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.logdir, 'logging.log'),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG
    )

    # Print result directory
    print(f"Result directory: {args.logdir}")
    logging.info(f"Result directory: {args.logdir}")

    # Save config
    config_path = os.path.join(args.logdir, config_filename)
    yaml.dump(yaml.safe_load(OmegaConf.to_yaml(cfg)), open(config_path, 'w'))

    print(f"Config saved as {config_path}")
    logging.info(f"Config saved as {config_path}")

    main(args, cfg)
