import os
import lmdb
import numpy as np
import torch
import torchvision.utils as tvu
import torch.distributed as dist
import wandb

def save_imagenet_result(x, y, info, samples_root, suffix=""):
        
    if len(x.shape) == 3:
        n=1
    else:
        n = x.size(0)

        
    for i in range(n):
        #print('info["class_id"][i]', info["class_id"][i])
        class_dir = os.path.join(samples_root, info["class_id"][i])
        #print('class_dir', class_dir)
        os.makedirs(class_dir, exist_ok=True)
    for i in range(n):
        if len(suffix) > 0:
            tvu.save_image(x[i], os.path.join(samples_root, info["class_id"][i], f'{info["name"][i]}_{suffix}.png'))
        else:
            tvu.save_image(x[i], os.path.join(samples_root, info["class_id"][i], f'{info["name"][i]}.png'))

    # for i in range(n):
    #     if len(suffix) > 0:
    #         tvu.save_image(x[i], os.path.join(samples_root, f'{info["name"][i]}_{suffix}.png'))
    #     else:
    #         tvu.save_image(x[i], os.path.join(samples_root, f'{info["name"][i]}.png'))

    # dist.barrier()


# def save_ffhq_result(x, y, info, samples_root, suffix=""):
#     x_list = [torch.zeros_like(x) for i in range(dist.get_world_size())]
#     idx = info['index']
#     idx_list = [torch.zeros_like(idx) for i in range(dist.get_world_size())]
#     dist.gather(x, x_list, dst=0)
#     dist.gather(idx, idx_list, dst=0)

#     if len(suffix) == 0:
#         lmdb_path = f'{samples_root}.lmdb'
#     else:
#         lmdb_path = f'{samples_root}_{suffix}.lmdb'

#     lmdb_dir = lmdb_path.split('/')[:-1]
#     if len(lmdb_dir) > 0:
#         lmdb_dir = '/'.join(lmdb_dir)
#         os.makedirs(lmdb_dir, exist_ok=True)

#     if dist.get_rank() == 0:
#         x = torch.cat(x_list, dim=0).permute(0, 2, 3, 1).detach().cpu().numpy()
#         idx = torch.cat(idx_list, dim=0).detach().cpu().numpy()
#         x = (x * 255.).astype(np.uint8)
#         n = x.shape[0]
#         env = lmdb.open(lmdb_path, map_size=int(1e12), readonly=False)
#         with env.begin(write=True) as txn:
#             for i in range(n):
#                 xi = x[i].copy()
#                 txn.put(str(int(idx[i])).encode(), xi)

#     dist.barrier()

def save_ffhq_result(x, y, info, samples_root, suffix=""):
        
    if len(x.shape) == 3:
        n=1
    else:
        n = x.size(0)

    for i in range(n):
        if len(suffix) > 0:
            tvu.save_image(x[i], os.path.join(samples_root, f'sample_{suffix}_{info["index"][i]}.png'))
        else:
            tvu.save_image(x[i], os.path.join(samples_root, f'sample_{info["index"][i]}.png'))

    dist.barrier()

def save_trajectory(xts, samples_root, suffix=""):
    n = len(xts)
    for i in range(n):
        tvu.save_image(xts[i], os.path.join(samples_root, f'sample_{suffix}_{dist.get_rank()}_{i}.png'))

    dist.barrier()


def save_result(name, x, y, info, samples_root, suffix=""):
    if 'ImageNet' in name:
        save_imagenet_result(x, y, info, samples_root, suffix)
    elif 'FFHQ' in name:
        save_ffhq_result(x, y, info, samples_root, suffix)
    elif 'kodak' in name:
        save_ffhq_result(x, y, info, samples_root, suffix)
    elif 'CelebAHQ' in name:
        save_ffhq_result(x, y, info, samples_root, suffix)
    elif 'LSUN' in name:
        save_ffhq_result(x, y, info, samples_root, suffix)

def log_wandb(
        self,
        vectors,
        labels,
        resize_shape=None,
        normalize=True,
        caption="Comparison",
        **kwargs,
    ):
        """
        Args:
            vectors (list of torch.Tensor): List of tensors to be visualized. Each tensor should be 4D (N, C, H, W).
            labels (list of str): List of labels corresponding to each tensor.
            resize_shape (tuple, optional): Target size (H, W) for resizing tensors. Defaults to None.
            normalize (bool, optional): Whether to normalize tensors to [0, 1]. Defaults to True.
            caption (str, optional): Caption for the WandB log. Defaults to "Comparison".
            kwargs: Additional arguments passed to WandB logging.
        """
        # TODO: This should not be sampler specific. Refactor later!
        if kwargs.get("dist", None) and kwargs["dist"].get_rank() != 0:
            return

        if kwargs.get("use_wandb", False):
            processed_images = []

            for vector, label in zip(vectors, labels):
                if normalize:
                    vector = (vector - vector.min()) / (vector.max() - vector.min())
                if resize_shape:
                    if self.cfg.deg.name == "bicubic":
                        vector = F.interpolate(
                            vector, size=resize_shape, mode="bilinear", align_corners=False
                        )
                processed_image = (
                    vector[0].permute(1, 2, 0).detach().cpu().numpy()
                )  # HWC
                processed_images.append(
                    (processed_image * 255).astype("uint8")
                )  # [0, 255]

            concatenated_image = torch.cat(
                [torch.tensor(img) for img in processed_images], dim=1
            ).numpy()
            wandb.log(
                {caption: wandb.Image(concatenated_image, caption=" | ".join(labels))}
            )