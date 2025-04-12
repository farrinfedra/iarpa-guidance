import os
from typing import List, Optional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset
from PIL import Image


class CelebAHQDataset(Dataset):
    def __init__(self, root: str, transform: Optional[transforms.Compose] = None):
        self.root = root
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self) -> List[str]:
        # Combine male and female images
        image_paths = []
        for subdir in ["male", "female"]:
            subdir_path = os.path.join(self.root, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_paths.append(os.path.join(subdir_path, file))
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        target = 0
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")  # Convert to RGB if necessary

        if self.transform:
            image = self.transform(image)

        return image, target, {"index": index}


def get_celeba_hq_dataset(
    root: str,
    transform: Optional[str] = "default",
    image_size: int = 256,
    subset=-1,
    **kwargs
):
    """

    Args:
        root (str): Root directory of the dataset.
        transform (Optional[str]): Transformation to apply. Defaults to "default".
        image_size (int): Target image size for resizing/cropping. 256 in our exps.
    """
    if transform == "default":
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
    elif transform == "identity":
        transform = transforms.Compose([transforms.PILToTensor()])
    else:
        raise ValueError(f"Unsupported transform: {transform}")

    dset = CelebAHQDataset(root=root, transform=transform)
    if isinstance(subset, int) and subset > 0:
        dset = Subset(dset, list(range(subset)))
    else:
        assert isinstance(subset, list)
        dset = Subset(dset, subset)

    return dset


def get_celeba_hq_loader(
    dset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
    **kwargs
):
    sampler = DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last)
    loader = DataLoader(
        dset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    return loader
