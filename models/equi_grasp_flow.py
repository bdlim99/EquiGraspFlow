import torch
from copy import deepcopy

from utils.Lie import inv_SO3, log_SO3, exp_so3, bracket_so3


class EquiGraspFlow(torch.nn.Module):
    def __init__(self, p_uncond, guidance, init_dist, encoder, vector_field, ode_solver):
        super().__init__()

        self.p_uncond = p_uncond
        self.guidance = guidance

        self.init_dist = init_dist
        self.encoder = encoder
        self.vector_field = vector_field
        self.ode_solver = ode_solver

    def step(self, data, losses, split, optimizer=None):
        # Get data
        pc = data['pc']
        x_1 = data['Ts_grasp']

        # Get number of grasp poses in each batch and combine batched data
        nums_grasps = torch.tensor([len(Ts_grasp) for Ts_grasp in x_1], device=data['pc'].device)

        x_1 = torch.cat(x_1, dim=0)

        # Sample t and x_0
        t = torch.rand(len(x_1), 1).to(x_1.device)
        x_0 = self.init_dist(len(x_1), x_1.device)

        # Get x_t and u_t
        x_t, u_t = get_traj(x_0, x_1, t)

        # Forward
        v_t = self(pc, t, x_t, nums_grasps)

        # Calculate loss
        loss_mse = losses['mse'](v_t, u_t)

        loss = losses['mse'].weight * loss_mse

        # Backward
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        # Archive results
        results = {
            f'scalar/{split}/loss': loss.item(),
        }

        return results

    def forward(self, pc, t, x_t, nums_grasps):
        z = torch.zeros((len(pc), self.encoder.dims[-1], 3), device=pc.device)

        # Encode point cloud
        z = self.encoder(pc)

        # Repeat feature
        z = z.repeat_interleave(nums_grasps, dim=0)

        # Null condition
        mask_uncond = torch.bernoulli(torch.Tensor([self.p_uncond] * len(z))).to(bool)

        z[mask_uncond] = torch.zeros_like(z[mask_uncond])

        # Get vector
        v_t = self.vector_field(z, t, x_t)

        return v_t

    @torch.no_grad()
    def sample(self, pc, nums_grasps):
        # Sample initial samples
        x_0 = self.init_dist(sum(nums_grasps), pc.device)
        self.X0SAMPLED = deepcopy(x_0)

        # Encode point cloud
        z = self.encoder(pc)

        # Repeat feature
        z = z.repeat_interleave(nums_grasps, dim=0)

        # Push-forward initial samples
        x_1_hat = self.ode_solver(z, x_0, self.guided_vector_field)[:, -1]

        # Batch x_1_hat
        x_1_hat = x_1_hat.split(nums_grasps.tolist())

        return x_1_hat

    def guided_vector_field(self, z, t, x_t):
        v_t = (1 - self.guidance) * self.vector_field(torch.zeros_like(z), t, x_t) + self.guidance * self.vector_field(z, t, x_t)

        return v_t


def get_traj(x_0, x_1, t):
    # Get rotations
    R_0 = x_0[:, :3, :3]
    R_1 = x_1[:, :3, :3]

    # Get translations
    p_0 = x_0[:, :3, 3]
    p_1 = x_1[:, :3, 3]

    # Get x_t
    x_t = torch.eye(4).repeat(len(x_1), 1, 1).to(x_1)
    x_t[:, :3, :3] = (R_0 @ exp_so3(t.unsqueeze(2) * log_SO3(inv_SO3(R_0) @ R_1)))
    x_t[:, :3, 3] = p_0 + t * (p_1 - p_0)

    # Get u_t
    u_t = torch.zeros(len(x_1), 6).to(x_1)
    u_t[:, :3] = bracket_so3(log_SO3(inv_SO3(R_0) @ R_1))
    u_t[:, :3] = torch.einsum('bij,bj->bi', R_0, u_t[:, :3])    # Convert w_b to w_s
    u_t[:, 3:] = p_1 - p_0

    return x_t, u_t
