import numpy as np
import torch


def point2result(points, labels, num_classes):
    if points.shape[0] == 0:
        return [np.zeros((0, 3), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [points[labels == i, :] for i in range(num_classes)]
