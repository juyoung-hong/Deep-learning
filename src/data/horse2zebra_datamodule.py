# https://github.com/ndb796/PyTorch-Image-to-Image-Translation/blob/main/CycleGAN_for_Horse2Zebra_Translation.ipynb
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import os
import random

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms_

        self.files_A = sorted(f for f in (Path(root) / mode / 'A').glob('*') if f.suffix == '.jpg')
        self.files_B = sorted(f for f in (Path(root) / mode / 'B').glob('*') if f.suffix == '.jpg')

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]) 

        if img_A.mode != "RGB":
            img_A = img_A.convert('RGB')
        if img_B.mode != "RGB":
            img_B = img_B.convert('RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class Horse2ZebraDataModule:
    """DataModule for CIFAR10 dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        size: List = [64, 64],
        num_workers: int = 0,
        pin_memory: bool = False,
        worker_init_fn: Callable = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.Resize([int(s*1.12) for s in size]),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

        if not os.path.exists(f'{self.data_dir}horse2zebra.zip'):
            os.system(f'wget http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip -O {self.data_dir}horse2zebra.zip')
            os.system(f'unzip {self.data_dir}horse2zebra.zip -d {self.data_dir}')
            
            os.system(f'mkdir -p {self.data_dir}horse2zebra/train')
            os.system(f'mkdir -p {self.data_dir}horse2zebra/test')

            os.system(f'mv {self.data_dir}horse2zebra/trainA {self.data_dir}horse2zebra/train/A')
            os.system(f'mv {self.data_dir}horse2zebra/trainB {self.data_dir}horse2zebra/train/B')
            os.system(f'mv {self.data_dir}horse2zebra/testA {self.data_dir}horse2zebra/test/A')
            os.system(f'mv {self.data_dir}horse2zebra/testB {self.data_dir}horse2zebra/test/B')

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            self.data_train = ImageDataset(f'{self.data_dir}horse2zebra', transforms_=self.transforms, mode='train')
            self.data_test = ImageDataset(f'{self.data_dir}horse2zebra', transforms_=self.transforms, mode='test')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            shuffle=False,
            drop_last=True,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = Horse2ZebraDataModule()
