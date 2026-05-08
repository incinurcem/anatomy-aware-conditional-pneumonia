import numpy as np
import pytest

from pneumo_pipeline.cond.condition_vector import build_condition_vector


def test_condition_vector_shape():
    """
    Condition vector doğru boyutta mı kontrol edilir.
    """
    cond = build_condition_vector(
        pneumonia_prob=0.7,
        burden=0.3,
        bilateral=True,
        qc_score=0.9,
        uncertainty=0.2
    )

    assert isinstance(cond, np.ndarray)
    assert cond.shape[0] == 5


def test_condition_vector_range():
    """
    Condition değerleri 0-1 aralığında olmalı.
    """
    cond = build_condition_vector(
        pneumonia_prob=0.5,
        burden=0.1,
        bilateral=False,
        qc_score=0.8,
        uncertainty=0.3
    )

    assert np.all(cond >= 0.0)
    assert np.all(cond <= 1.0)


def test_condition_vector_bilateral_flag():
    """
    Bilateral bool değeri doğru encode ediliyor mu?
    """
    cond_true = build_condition_vector(0.5, 0.2, True, 0.9, 0.1)
    cond_false = build_condition_vector(0.5, 0.2, False, 0.9, 0.1)

    assert cond_true[2] == 1.0
    assert cond_false[2] == 0.0


def test_condition_vector_invalid_input():
    """
    Geçersiz değerler hata üretmeli.
    """
    with pytest.raises(ValueError):
        build_condition_vector(
            pneumonia_prob=1.5,
            burden=0.3,
            bilateral=True,
            qc_score=0.9,
            uncertainty=0.2
        )