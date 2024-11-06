import roma
import torch


def get_dist(cfg):
    name = cfg.pop('name')

    if name == 'SO3_uniform_R3_normal':
        dist_fn = SO3_uniform_R3_normal
    elif name == 'SO3_uniform_R3_spherical':
        dist_fn = SO3_uniform_R3_spherical
    elif name == 'SO3_centripetal_R3_normal':
        dist_fn = SO3_centripetal_R3_normal
    elif name == 'SO3_centripetal_R3_spherical':
        dist_fn = SO3_centripetal_R3_spherical
    else:
        raise NotImplementedError(f"Distribution {name} is not implemented.")
    
    return dist_fn


def SO3_uniform_R3_normal(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = torch.randn(num_samples, 3).to(device)

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


def SO3_uniform_R3_spherical(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = torch.randn(num_samples, 3).to(device)
    p /= p.norm(dim=-1, keepdim=True)

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


def SO3_centripetal_R3_normal(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = - (0.112 * 5 + torch.randn(num_samples, 1).to(device).abs()) * R[:, :, 2]

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


def SO3_centripetal_R3_spherical(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = - R[:, :, 2]

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T
