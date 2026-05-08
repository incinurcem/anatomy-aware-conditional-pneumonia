import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import cv2
import torch
from torch.utils.data import Dataset

#d
# ---------------------------------------------------------
# Example Dataset Class (simplified version used for tests)
# ---------------------------------------------------------

class ClassifierDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row["image_id"]
        label = row["label"]

        image_path = os.path.join(self.image_dir, image_id)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Image could not be read: {image_path}")

        image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)

        image_tensor = torch.from_numpy(image)

        label_tensor = torch.tensor(label).long()

        return image_tensor, label_tensor


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def temp_dataset():

    with tempfile.TemporaryDirectory() as tmpdir:

        image_dir = tmpdir

        image_ids = []
        labels = []

        for i in range(5):

            img = np.random.randint(
                0, 255,
                (256, 256),
                dtype=np.uint8
            )

            img_name = f"img_{i}.png"

            img_path = os.path.join(image_dir, img_name)

            cv2.imwrite(img_path, img)

            image_ids.append(img_name)
            labels.append(i % 2)

        df = pd.DataFrame({
            "image_id": image_ids,
            "label": labels
        })

        yield df, image_dir


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_dataset_length(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    assert len(dataset) == len(df)


def test_dataset_returns_tensor(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_image_shape(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    image, _ = dataset[0]

    assert image.shape[0] == 1
    assert image.shape[1] == 256
    assert image.shape[2] == 256


def test_label_values(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    _, label = dataset[1]

    assert label.item() in [0, 1]


def test_multiple_samples_access(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    for i in range(len(dataset)):

        image, label = dataset[i]

        assert image.shape == (1, 256, 256)
        assert label.item() in [0, 1]


def test_invalid_image_path():

    df = pd.DataFrame({
        "image_id": ["missing.png"],
        "label": [1]
    })

    dataset = ClassifierDataset(df, "/tmp")

    with pytest.raises(ValueError):

        _ = dataset[0]


def test_dataset_tensor_range(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    image, _ = dataset[0]

    assert torch.max(image) <= 1.0
    assert torch.min(image) >= 0.0


def test_dataset_dtype(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    image, _ = dataset[0]

    assert image.dtype == torch.float32


def test_label_dtype(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    _, label = dataset[0]

    assert label.dtype == torch.int64


def test_dataset_iteration(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    for image, label in dataset:

        assert image.shape == (1, 256, 256)
        assert label.item() in [0, 1]


def test_dataframe_structure(temp_dataset):

    df, image_dir = temp_dataset

    assert "image_id" in df.columns
    assert "label" in df.columns


def test_dataset_indexing(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    image0, label0 = dataset[0]
    image1, label1 = dataset[1]

    assert not torch.equal(image0, image1) or label0 != label1


def test_dataset_random_access(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    idxs = [4, 2, 0, 3]

    for idx in idxs:

        image, label = dataset[idx]

        assert image.shape == (1, 256, 256)
        assert label.item() in [0, 1]


def test_dataset_len_matches_dataframe(temp_dataset):

    df, image_dir = temp_dataset

    dataset = ClassifierDataset(df, image_dir)

    assert len(dataset) == df.shape[0]