import torch
import numpy as np

from pneumo_pipeline.seg.dataset import LungSegDataset

#d
class DummyDataset(LungSegDataset):

    def __init__(self):
        self.images = [np.random.rand(224,224) for _ in range(5)]
        self.masks = [np.random.rand(224,224) for _ in range(5)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = torch.tensor(self.images[idx]).float().unsqueeze(0)
        mask = torch.tensor(self.masks[idx]).float().unsqueeze(0)

        return img, mask


def test_dataset_length():

    dataset = DummyDataset()

    assert len(dataset) == 5


def test_dataset_item():

    dataset = DummyDataset()

    img, mask = dataset[0]

    assert img.shape == (1, 224, 224)
    assert mask.shape == (1, 224, 224)


def test_dataset_tensor():

    dataset = DummyDataset()

    img, mask = dataset[1]

    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)