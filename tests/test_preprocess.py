import numpy as np
from pneumo_pipeline.preprocess.preprocess_pipeline import (
    normalize_to_float01,
    resize_image,
)

#d
def test_resize_image():
    """
    Görüntü doğru boyuta resize ediliyor mu
    """

    img = np.random.rand(512, 512)

    resized = resize_image(img, size=(224, 224))

    assert resized.shape == (224, 224)


def test_normalization_range():
    """
    Normalize sonrası değerler 0-1 aralığında mı
    """

    img = np.random.randint(0, 255, (224, 224))

    norm = normalize_image(img)

    assert norm.min() >= 0
    assert norm.max() <= 1


def test_preprocess_pipeline():
    """
    Resize + normalize birlikte çalışıyor mu
    """

    img = np.random.randint(0, 255, (512, 512))

    resized = resize_image(img, size=224)
    norm = normalize_image(resized)

    assert norm.shape == (224, 224)