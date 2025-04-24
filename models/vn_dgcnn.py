import torch

from models.vn_layers import VNLinearLeakyReLU, knn


class VNDGCNNEncoder(torch.nn.Module):
    def __init__(self, num_neighbors, dims=[1, 21, 21, 42, 85, 341], use_bn=False):
        super().__init__()

        self.num_neighbors = num_neighbors
        self.dims = dims

        layers = []

        for dim_in, dim_out in zip(dims[:-2], dims[1:-1]):
            layers += [VNLinearLeakyReLU(2 * dim_in, dim_out, use_bn=use_bn)]

        layers += [VNLinearLeakyReLU(sum(dims[1:-1]), dims[-1], dim=4, share_nonlinearity=True, use_bn=use_bn)]

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_list = []

        for layer in self.layers[:-1]:
            x = get_graph_feature(x, k=self.num_neighbors)
            x = layer(x)
            x = x.mean(dim=-1)

            x_list += [x]

        x = torch.cat(x_list, dim=1)

        x = self.layers[-1](x)
        x = x.mean(dim=-1)

        return x


def get_graph_feature(x, k=20):
    batch_size = x.shape[0]
    num_pts = x.shape[3]

    x = x.view(batch_size, -1, num_pts)

    idx = knn(x, k=k)
    idx_base = torch.arange(0, batch_size, device=idx.device).unsqueeze(1).unsqueeze(2) * num_pts
    idx = idx + idx_base
    idx = idx.view(-1)

    num_dims = x.shape[1] // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_pts, -1)[idx]
    feature = feature.view(batch_size, num_pts, k, num_dims, 3)
    x = x.view(batch_size, num_pts, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature