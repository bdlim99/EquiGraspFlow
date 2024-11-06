import torch

from models.vn_mlp import VNMLPwithScalar


class VectorField4SE3(VNMLPwithScalar):
    def forward(self, z, t, x_t):
        # Construct scalar-list and vector-list
        s = t.unsqueeze(1)
        v = torch.cat((z, x_t[:, :3].transpose(1, 2)), dim=1)

        # Forward VN-MLP with scalar-list
        out = self._forward(s, v)

        # Convert two 3-dim vectors to one 6-dim vector
        out = out.view(-1, 6)

        return out
