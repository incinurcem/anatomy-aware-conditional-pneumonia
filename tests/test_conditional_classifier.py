import torch
import pytest
#d
from pneumo_pipeline.cls.models.conditional_classifier import ConditionalClassifier


def test_classifier_forward():
    """
    Model forward pass çalışıyor mu?
    """

    model = ConditionalClassifier(
        in_channels=1,
        cond_dim=5
    )

    x = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 5)

    out = model(x, cond)

    assert out.shape == (2, 1)


def test_classifier_output_range():
    """
    Sigmoid çıktısı 0-1 aralığında mı?
    """

    model = ConditionalClassifier(
        in_channels=1,
        cond_dim=5
    )

    x = torch.randn(4, 1, 224, 224)
    cond = torch.randn(4, 5)

    out = model(x, cond)

    assert torch.min(out) >= 0
    assert torch.max(out) <= 1


def test_classifier_batch():
    """
    Batch dimension korunuyor mu?
    """

    model = ConditionalClassifier(
        in_channels=1,
        cond_dim=5
    )

    batch_size = 8

    x = torch.randn(batch_size, 1, 224, 224)
    cond = torch.randn(batch_size, 5)

    out = model(x, cond)

    assert out.shape[0] == batch_size