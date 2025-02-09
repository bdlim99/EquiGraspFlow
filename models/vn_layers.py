import torch


EPS = 1e-6


class VNLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.map_to_feat = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)

        return x_out


class VNBatchNorm(torch.nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()

        if dim == 3 or dim == 4:
            self.bn = torch.nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = torch.nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNLinearLeakyReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_bn=True, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope
        self.use_bn = use_bn

        # Linear
        self.map_to_feat = torch.nn.Linear(in_channels, out_channels, bias=False)

        # BatchNorm
        if use_bn:
            self.bn = VNBatchNorm(out_channels, dim=dim)

        # LeakyReLU
        if share_nonlinearity:
            self.map_to_dir = torch.nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)

        # BatchNorm
        if self.use_bn:
            p = self.bn(p)

        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))

        return x_out


def knn(x, k):
    pairwise_distance = (x.unsqueeze(-1) - x.unsqueeze(-2)).norm(dim=1) ** 2

    idx = pairwise_distance.topk(k, dim=-1, largest=False)[1]   # (batch_size, num_pts, k)

    return idx