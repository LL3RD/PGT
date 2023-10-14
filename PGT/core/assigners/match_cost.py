from mmdet.core.bbox.match_costs.builder import MATCH_COST
import torch


@MATCH_COST.register_module()
class PointL1Cost:

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, point_pred, gt_points):
        point_cost = torch.cdist(point_pred, gt_points, p=1)
        return point_cost * self.weight
