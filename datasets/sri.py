import os
from typing import List, Optional, Any, Tuple
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
from functools import partial
import torch
import numpy as np
from torchvision.transforms import InterpolationMode

IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm",
    ".tif", ".tiff", ".webp", ".JPEG",
)

def _is_image_file(p: str) -> bool:
    return any(p.endswith(ext) for ext in IMG_EXTENSIONS)

def center_crop_arr(pil_image, image_size=256):
    """
    OpenAI guided-diffusion style: progressive BOX downsample (if needed),
    BICUBIC resize to make min side = image_size, then center-crop to (image_size, image_size).
    Returns a numpy array (H, W, C).
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def get_transform(
    transform_type: str,
    orig_image_size: Optional[Tuple[int, int]] = None,
    image_size: int = 256,
):
    """
    transform_type:
      - "diffusion"
      - "sri"  (center-square -> Resize to image_size with LANCZOS+antialias -> ToTensor)
    """
    if transform_type == "diffusion":
        return transforms.Compose([
            partial(center_crop_arr, image_size=image_size),
            transforms.ToTensor(),
        ])

    elif transform_type == "sri":
        # If orig_image_size is given, use that fixed size for CenterCrop.
        # Otherwise, compute per-image with a Lambda.
        if orig_image_size is not None:
            min_side = min(orig_image_size)
            center_square = transforms.CenterCrop(min_side)
        else:
            center_square = transforms.Lambda(
                lambda im: transforms.functional.center_crop(im, min(im.size))
            )

        return transforms.Compose([
            center_square,
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.LANCZOS,
                antialias=True,
            ),
            transforms.ToTensor(),
        ])

    else:
        raise ValueError(f"Unsupported transform type: {transform_type}")

class SRIDataset(Dataset):
    """
    Expects `root` to point to a directory that directly contains image files.
    Returns: (image_tensor, target, meta_dict)
    """
    def __init__(self, root: str, transform: Optional[transforms.Compose] = None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self) -> List[str]:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")
        files = [f for f in os.listdir(self.root) if _is_image_file(f)]
        files.sort()
        return [os.path.join(self.root, f) for f in files]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Any, int, dict]:
        target = 0
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return image, target, {"index": index, "path": image_path, "name": stem}

def get_sri_dataset(
    *,
    root: str,
    image_size: int = 256,
    transform: str = "sri",
    orig_image_size: Optional[Tuple[int, int]] = None,  # <-- pass here if fixed
) -> SRIDataset:
    """
    root: directory containing images
    orig_image_size: set to (4032, 3024) if all images share that size; else leave None.
    """
    t = get_transform(transform, orig_image_size=orig_image_size, image_size=image_size)
    return SRIDataset(root=root, transform=t)

def get_sri_loader(
    dset: Dataset,
    *,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
    distributed: bool = False,
) -> DataLoader:
    sampler = DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last) if distributed else None
    loader = DataLoader(
        dset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return loader
