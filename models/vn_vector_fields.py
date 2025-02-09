import torch

from models.vn_layers import VNLinearLeakyReLU, VNLinear


class VNVectorFields(torch.nn.Module):
    def __init__(self, dims, use_bn):
        super().__init__()

        # Setup lifting layer
        self.lifting_layer = VNLinear(dims[0] - 1, 1)

        # Setup VN-MLP
        layers = []

        for i in range(len(dims)-2):
            layers += [VNLinearLeakyReLU(dims[i], dims[i+1], dim=4, use_bn=use_bn)]

        layers += [VNLinear(dims[-2], dims[-1])]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, z, t, x_t):
        # Construct scalar-list and vector-list
        s = t.unsqueeze(1)
        v = torch.cat((z, x_t[:, :3].transpose(1, 2)), dim=1)

        # Lift scalar-list to vector-list
        trans = self.lifting_layer(v)
        v_s = s @ trans

        # Concatenate
        v = torch.cat((v, v_s), dim=1)

        # Forward VN-MLP
        out = self.layers(v).contiguous()

        # Convert two 3-dim vectors to one 6-dim vector
        out = out.view(-1, 6)

        return out
