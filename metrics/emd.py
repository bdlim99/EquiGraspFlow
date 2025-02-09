from scipy.optimize import linear_sum_assignment

from utils.Lie import SE3_geodesic_dist


class EMDCalculator:
    def __init__(self, type):
        self.type = type

    def calculate_distance(self, x, y):
        if self.type == 'SE3':
            T_x = x.view(-1, 4, 4)
            T_y = y.view(-1, 4, 4)

            return SE3_geodesic_dist(T_x, T_y).view(x.shape[:2])
        elif self.type == 'L2':
            return ((x - y) ** 2).sum(dim=3).sqrt()
        else:
            raise NotImplementedError(f"Type {self.type} is not implemented. Choose type between 'SE3' and 'L2'.")

    def __call__(self, source, target):
        assert len(source) == len(target), f"The number of samples in source {len(source)} must be equal to the number of samples in target {len(target)}."

        source = source.unsqueeze(1).repeat(1, len(target), 1, 1)
        target = target.unsqueeze(0).repeat(len(source), 1, 1, 1)

        distance = self.calculate_distance(source, target).cpu().numpy()

        idxs_row, idxs_col = linear_sum_assignment(distance)

        emd = distance[idxs_row, idxs_col].mean()

        return emd