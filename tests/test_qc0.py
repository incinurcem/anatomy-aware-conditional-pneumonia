import numpy as np

from pneumo_pipeline.qc.qc0 import compute_qc_score

#d
def test_qc_range():
    """
    QC skoru 0-1 aralığında olmalı
    """

    img = np.random.rand(224, 224)

    qc = compute_qc_score(img)

    assert 0 <= qc <= 1


def test_qc_high_quality():
    """
    Temiz görüntü yüksek QC üretmeli
    """

    img = np.ones((224, 224)) * 0.8

    qc = compute_qc_score(img)

    assert qc > 0.5


def test_qc_low_quality():
    """
    Gürültülü görüntü düşük QC üretmeli
    """

    img = np.random.rand(224, 224) * 0.1

    qc = compute_qc_score(img)

    assert qc < 0.8