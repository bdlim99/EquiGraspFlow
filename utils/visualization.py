from plotly.subplots import make_subplots
from plotly import graph_objects as go
import numpy as np
import torch


class PlotlySubplotsVisualizer:
    def __init__(self, rows, cols):
        self.num_subplots = rows * cols

        self.reset(rows, cols)

    def reset(self, rows, cols):
        self.fig = make_subplots(rows=rows, cols=cols, specs=[[{'is_3d': True}]*cols]*rows)
        self.fig.update_layout(height=900)

    def add_vector(self, x, y, z, u, v, w, row, col, color='black', width=5, sizeref=0.2, showlegend=False):
        self.fig.add_trace(
            go.Scatter3d(x=[x, x+0.9*u], y=[y, y+0.9*v], z=[z, z+0.9*w], mode='lines', line=dict(color=color, width=width), showlegend=showlegend),
            row=row, col=col
        )
        self.fig.add_trace(
            go.Cone(x=[x+u], y=[y+v], z=[z+w], u=[u], v=[v], w=[w], sizemode='absolute', sizeref=sizeref, anchor='tip', colorscale=[[0, color], [1, color]], showscale=False),
            row=row, col=col
        )

    def add_mesh(self, mesh, row, col, color='aquamarine', idx='', showlegend=False):
        vertices = np.asarray(mesh.vertices)
        traingles = np.asarray(mesh.triangles)

        self.fig.add_trace(
            go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=traingles[:, 0], j=traingles[:, 1], k=traingles[:, 2], color=color, showlegend=showlegend, name='mesh '+str(idx)),
            row=row, col=col
        )

    def add_pc(self, pc, row, col, color='lightpink', size=5, idx='', showlegend=False):
        self.fig.add_trace(
            go.Scatter3d(x=pc[:,0], y=pc[:,1], z=pc[:,2], mode='markers', marker=dict(size=size, color=color), showlegend=showlegend, name='pc '+str(idx)),
            row=row, col=col,
        )

    def add_gripper(self, T, row, col, color='violet', width=5, idx='', showlegend=False):
        gripper_scatter3d = get_gripper_scatter3d(T, color, width, idx, showlegend)

        self.fig.add_trace(gripper_scatter3d, row=row, col=col)

    def add_grippers(self, Ts, row, col, color='violet', width=5, idx='', showlegend=False):
        for T_gripper in Ts:
            gripper_scatter3d = get_gripper_scatter3d(T_gripper, color, width, idx, showlegend)

            self.fig.add_trace(gripper_scatter3d, row=row, col=col)

    def add_frame(self, T, row, col, size=1, width=5, sizeref=0.2):
        self.add_vector(T[0, 3], T[1, 3], T[2, 3], size * T[0, 0], size * T[1, 0], size * T[2, 0], row, col, color='red', width=width, sizeref=sizeref)
        self.add_vector(T[0, 3], T[1, 3], T[2, 3], size * T[0, 1], size * T[1, 1], size * T[2, 1], row, col, color='green', width=width, sizeref=sizeref)
        self.add_vector(T[0, 3], T[1, 3], T[2, 3], size * T[0, 2], size * T[1, 2], size * T[2, 2], row, col, color='blue', width=width, sizeref=sizeref)


def get_gripper_scatter3d(T, color, width=5, idx='', showlegend=False):
    unit1 = 0.066 #* 8 # 0.56
    unit2 = 0.041 #* 8 # 0.32
    unit3 = 0.046 #* 8 # 0.4

    pbase = torch.Tensor([0, 0, 0, 1]).reshape(1, -1)
    pcenter = torch.Tensor([0, 0, unit1, 1]).reshape(1, -1)
    pleft = torch.Tensor([unit2, 0, unit1, 1]).reshape(1, -1)
    pright = torch.Tensor([-unit2, 0, unit1, 1]).reshape(1, -1)
    plefttip = torch.Tensor([unit2, 0, unit1+unit3, 1]).reshape(1, -1)
    prighttip = torch.Tensor([-unit2, 0, unit1+unit3, 1]).reshape(1, -1)

    hand = torch.cat([pbase, pcenter, pleft, pright, plefttip, prighttip], dim=0).to(T)
    hand = torch.einsum('ij, kj -> ik', T, hand).cpu()

    phandx = [hand[0, 4], hand[0, 2], hand[0, 1], hand[0, 0], hand[0, 1], hand[0, 3], hand[0, 5]]
    phandy = [hand[1, 4], hand[1, 2], hand[1, 1], hand[1, 0], hand[1, 1], hand[1, 3], hand[1, 5]]
    phandz = [hand[2, 4], hand[2, 2], hand[2, 1], hand[2, 0], hand[2, 1], hand[2, 3], hand[2, 5]]

    gripper_scatter3d = go.Scatter3d(x=phandx, y=phandy, z=phandz, mode='lines', line=dict(color=color, width=width), showlegend=showlegend, name='gripper '+str(idx))

    return gripper_scatter3d
