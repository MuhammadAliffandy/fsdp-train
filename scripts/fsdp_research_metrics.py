"""Physical native-grid boundary diagnostics for selected research predictions."""
import numpy as np
import torch


def surface_metrics(predicted, target, shape, spacing):
    from monai.metrics import HausdorffDistanceMetric, SurfaceDiceMetric
    predicted, target = np.unique(predicted), np.unique(target)
    if not len(target):
        raise ValueError('FSDP surface metrics require a nonempty native reference')
    if not len(predicted):
        return {'hd95_mm': None, 'surface_dice_0_5mm': 0., 'surface_dice_1mm': 0., 'empty_prediction': True}
    combined = np.union1d(predicted, target)
    coordinates = np.stack(np.unravel_index(combined, shape), 1)
    start = np.maximum(0, coordinates.min(0) - 1)
    stop = np.minimum(shape, coordinates.max(0) + 2)
    masks = []
    for indices in (predicted, target):
        points = np.stack(np.unravel_index(indices, shape), 1) - start
        mask = np.zeros(tuple(stop - start), dtype=np.uint8)
        mask[tuple(points.T)] = 1
        masks.append(torch.from_numpy(mask)[None, None])
    hd95 = HausdorffDistanceMetric(include_background=True, percentile=95, directed=False,
                                   reduction='none')(*masks, spacing=spacing).item()
    surfaces = [SurfaceDiceMetric([tolerance], include_background=True, use_subvoxels=True,
                                  reduction='none')(*masks, spacing=spacing).item() for tolerance in (.5, 1.)]
    if not np.isfinite([hd95, *surfaces]).all():
        raise ValueError('Nonfinite native surface metric')
    return {'hd95_mm': float(hd95), 'surface_dice_0_5mm': float(surfaces[0]),
            'surface_dice_1mm': float(surfaces[1]), 'empty_prediction': False}
