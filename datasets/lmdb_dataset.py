import io
import os

import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image


class LMDBDataset(data.Dataset):
    def __init__(
        self,
        root,
        name="",
        split="train",
        transform=None,
        decode_key=False,
        is_encoded=False,
    ):
        self.name = name
        self.transform = transform
        self.split = split

        if self.split == "train":
            lmdb_path = os.path.join(root, "train.lmdb")
        elif self.split == "val":
            lmdb_path = os.path.join(root, "validation.lmdb")
        else:
            lmdb_path = os.path.join(f"{root}.lmdb")

        self.data_lmdb = lmdb.open(
            lmdb_path,
            readonly=True,
            max_readers=1,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.is_encoded = is_encoded
        self.decode_key = decode_key

        if self.decode_key:
            self.keys = []
            with self.data_lmdb.begin(write=False) as txn:
                for key, _ in txn.cursor():
                    self.keys.append(key)

    def __getitem__(self, index):
        target = 0
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            if self.decode_key:  # decoded key for img from memory
                key = self.keys[index]
            else:
                key = str(index).encode()

            data = txn.get(key)
            if data is None:
                raise ValueError(
                    f"Key {key.decode('ascii') if self.decode_key else key} not found in LMDB database."
                )

            if self.is_encoded:
                img = Image.open(io.BytesIO(data)).convert("RGB")

            else:
                img = np.asarray(data, dtype=np.uint8)
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, target, {"index": index}

    def __len__(self):
        if hasattr(self, "length"):
            return self.length
        else:
            with self.data_lmdb.begin() as txn:
                self.length = txn.stat()["entries"]
            return self.length
