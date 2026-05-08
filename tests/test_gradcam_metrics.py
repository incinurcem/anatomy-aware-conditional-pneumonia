import numpy as np

from pneumo_pipeline.explain.gradcam_metrics import compute_iou
from pneumo_pipeline.explain.gradcam_metrics import compute_hit_rate
#d

def test_iou_perfect():
    """
    Aynı maskelerde IoU = 1 olmalı
    """

    mask = np.ones((10, 10))

    iou = compute_iou(mask, mask)

    assert iou == 1.0


def test_iou_zero():
    """
    Çakışma yoksa IoU = 0
    """

    mask1 = np.zeros((10, 10))
    mask2 = np.ones((10, 10))

    iou = compute_iou(mask1, mask2)

    assert iou == 0.0


def test_hit_rate_true():
    """
    GradCAM bbox içine giriyorsa hit
    """

    cam = np.zeros((10, 10))
    cam[5, 5] = 1

    bbox = [4, 4, 6, 6]

    hit = compute_hit_rate(cam, bbox)

    assert hit == 1


def test_hit_rate_false():
    """
    GradCAM bbox dışında ise miss
    """

    cam = np.zeros((10, 10))
    cam[0, 0] = 1

    bbox = [4, 4, 6, 6]

    hit = compute_hit_rate(cam, bbox)

    assert hit == 0