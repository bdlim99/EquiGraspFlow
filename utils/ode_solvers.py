import torch
from copy import deepcopy

from utils.Lie import bracket_so3, exp_so3, Lie_bracket


def get_ode_solver(cfg):
    name = cfg.pop('name')

    if name == 'SE3_Euler':
        solver = SE3_Euler(**cfg)
    elif name == 'SE3_RK_mk':
        solver = SE3_RK4_MK(**cfg)
    else:
        raise NotImplementedError(f"ODE solver {name} is not implemented.")

    return solver


class SE3_Euler:
    def __init__(self, num_steps):
        self.t = torch.linspace(0, 1, num_steps + 1)

    @torch.no_grad()
    def __call__(self, z, x_0, func):
        # Initialize
        t = self.t.to(z.device)
        dt = t[1:] - t[:-1]
        traj = x_0.new_zeros(x_0.shape[0:1] + t.shape + x_0.shape[1:])
        traj[:, 0] = x_0

        for n in range(len(t)-1):
            # Get n-th values
            x_n = traj[:, n].contiguous()
            t_n = t[n].repeat(len(x_0), 1)
            h = dt[n].repeat(len(x_0), 1)

            ##### Stage 1 #####
            # Set function input
            x_hat = deepcopy(x_n)

            # Get vector (V_s)
            V_1 = func(z, t_n, x_hat)
            w_1 = V_1[:, :3]
            v_1 = V_1[:, 3:]

            # Change w_s to w_b and transform to matrix
            w_1 = torch.einsum('bji,bj->bi', x_hat[:, :3, :3], w_1)
            w_1 = bracket_so3(w_1)

            ##### Update #####
            traj[:, n+1] = deepcopy(x_n)
            traj[:, n+1, :3, :3] @= exp_so3(h.unsqueeze(-1) * w_1)
            traj[:, n+1, :3, 3] += h * v_1

        return traj


class SE3_RK4_MK:
    def __init__(self, num_steps):
        self.t = torch.linspace(0, 1, num_steps + 1)

    @torch.no_grad()
    def __call__(self, z, x_0, func):
        # Initialize
        t = self.t.to(z.device)
        dt = t[1:] - t[:-1]
        traj = x_0.new_zeros(x_0.shape[0:1] + t.shape + x_0.shape[1:])
        traj[:, 0] = x_0

        for n in range(len(t)-1):
            # Get n-th values
            x_n = traj[:, n].contiguous()
            t_n = t[n].repeat(len(x_0), 1)
            h = dt[n].repeat(len(x_0), 1)

            ##### Stage 1 #####
            # Set function input
            x_hat_1 = x_n

            # Get vector (V_s)
            V_1 = func(z, t_n, x_hat_1)
            w_1 = V_1[:, :3]
            v_1 = V_1[:, 3:]

            # Change w_s to w_b and transform to matrix
            w_1 = torch.einsum('bji,bj->bi', x_hat_1[:, :3, :3], w_1)
            w_1 = bracket_so3(w_1)

            # Set I_1
            I_1 = w_1

            ##### Stage 2 #####
            u_2 = h.unsqueeze(-1) * (1 / 2) * w_1
            u_2 += (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_2)

            # Set function input
            x_hat_2 = deepcopy(x_n)
            x_hat_2[:, :3, :3] @= exp_so3(u_2)
            x_hat_2[:, :3, 3] += h * (v_1 / 2)

            # Get vector (V_s)
            V_2 = func(z, t_n + (h / 2), x_hat_2)
            w_2 = V_2[:, :3]
            v_2 = V_2[:, 3:]

            # Change w_s to w_b and transform to matrix
            w_2 = torch.einsum('bji,bj->bi', x_hat_2[:, :3, :3], w_2)
            w_2 = bracket_so3(w_2)

            ##### Stage 3 #####
            u_3 = h.unsqueeze(-1) * (1 / 2) * w_2
            u_3 += (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_3)

            # Set function input
            x_hat_3 = deepcopy(x_n)
            x_hat_3[:, :3, :3] @= exp_so3(u_3)
            x_hat_3[:, :3, 3] += h * (v_2 / 2)

            # Get vector (V_s)
            V_3 = func(z, t_n + (h / 2), x_hat_3)
            w_3 = V_3[:, :3]
            v_3 = V_3[:, 3:]

            # Change w_s to w_b and transform to matrix
            w_3 = torch.einsum('bji,bj->bi', x_hat_3[:, :3, :3], w_3)
            w_3 = bracket_so3(w_3)

            ##### Stage 4 #####
            u_4 = h.unsqueeze(-1) * w_3
            u_4 += (h.unsqueeze(-1) / 6) * Lie_bracket(I_1, u_4)

            # Set function input
            x_hat_4 = deepcopy(x_n)
            x_hat_4[:, :3, :3] @= exp_so3(u_4)
            x_hat_4[:, :3, 3] += h * v_3

            # Get vector (V_s)
            V_4 = func(z, t_n + h, x_hat_4)
            w_4 = V_4[:, :3]
            v_4 = V_4[:, 3:]

            # Change w_s to w_b and transform to matrix
            w_4 = torch.einsum('bji,bj->bi', x_hat_4[:, :3, :3], w_4)
            w_4 = bracket_so3(w_4)

            ##### Update #####
            I_2 = (2 * (w_2 - I_1) + 2 * (w_3 - I_1) - (w_4 - I_1)) / h.unsqueeze(-1)
            u = h.unsqueeze(-1) * (1 / 6 * w_1 + 1 / 3 * w_2 + 1 / 3 * w_3 + 1 / 6 * w_4)
            u += (h.unsqueeze(-1) / 4) * Lie_bracket(I_1, u) + ((h ** 2).unsqueeze(-1) / 24) * Lie_bracket(I_2, u)

            traj[:, n+1] = deepcopy(x_n)
            traj[:, n+1, :3, :3] @= exp_so3(u)
            traj[:, n+1, :3, 3] += (h / 6) * (v_1 + 2 * v_2 + 2 * v_3 + v_4)

        return traj
