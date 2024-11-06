import torch

from models.vn_layers import VNLinearLeakyReLU, VNLinear


class VNMLPwithScalar(torch.nn.Module):
    def __init__(self, dim_scalar, dims, use_batchnorm):
        super().__init__()

        # Setup lifting layer
        self.lifting_layer = VNLinear(dims[0]-dim_scalar, 1)

        # Setup VN-MLP
        net = []

        for i in range(len(dims)-2):
            net += [VNLinearLeakyReLU(dims[i], dims[i+1], dim=4, use_batchnorm=use_batchnorm)]

        net += [VNLinear(dims[-2], dims[-1])]

        self.net = torch.nn.Sequential(*net)

    def _forward(self, s, v):
        # Lift scalar-list to vector-list
        trans = self.lifting_layer(v)
        v_s = s @ trans

        # Concatenate
        v = torch.cat((v, v_s), dim=1)

        # Forward VN-MLP
        out = self.net(v).contiguous()

        return out
