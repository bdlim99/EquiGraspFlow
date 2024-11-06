import torch

from models.vn_layers import VNLinearLeakyReLU, knn


class VNDGCNNEncoder(torch.nn.Module):
    def __init__(self, num_neighbors, dims = [1, 21, 21, 42, 85, 341], use_batchnorm=False):
        super().__init__()

        self.num_neighbors = num_neighbors
        self.dims = dims
        self.conv = []
        prev_dim = dims[0]
        for dim in dims[1:-1]:
            self.conv.append(VNLinearLeakyReLU(2 * prev_dim, dim, use_batchnorm=use_batchnorm))
            prev_dim = dim
        self.conv.append(VNLinearLeakyReLU(sum(dims[1:-1]), dims[-1], dim=4, share_nonlinearity=True, use_batchnorm=use_batchnorm))
        self.conv = torch.nn.ModuleList(self.conv)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_ = x
        xs = []
        for idx in range(len(self.conv)-1):
            x = get_graph_feature(x_, k=self.num_neighbors)
            x = self.conv[idx](x)
            x_ = x.mean(dim=-1)
            xs.append(x_)
        x = torch.cat(xs, dim=1)
        x = self.conv[-1](x)
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