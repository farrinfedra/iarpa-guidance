import hashlib
import os

import gdown
import requests
import torch.distributed as dist
from tqdm import tqdm

from .distributed import get_logger

URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
    "lsun_bedroom": "https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1",
    "ema_lsun_bedroom": "https://heibox.uni-heidelberg.de/f/b95206528f384185889b/?dl=1",
    "lsun_cat": "https://heibox.uni-heidelberg.de/f/fac870bd988348eab88e/?dl=1",
    "ema_lsun_cat": "https://heibox.uni-heidelberg.de/f/0701aac3aa69457bbe34/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50ef3c6436db52e/?dl=1",
    "imagenet_256_uncond": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
    "imagenet_256_cond": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt",
    "imagenet_256_classifier": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
    "imagenet_512_cond": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt",
    "imagenet_512_classifier": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt",
    "ffhq_256": "https://drive.google.com/uc\?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh", #not working TODO: update this link
    "celebahq": "https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&id=72419B431C262344%21103807&cid=72419B431C262344" #not working TODO: update this link
    
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
    "lsun_bedroom": "diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "ema_lsun_bedroom": "ema_diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "lsun_cat": "diffusion_lsun_cat_model/model-1761000.ckpt",
    "ema_lsun_cat": "ema_diffusion_lsun_cat_model/model-1761000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
    "ema_lsun_church": "ema_diffusion_lsun_church_model/model-4432000.ckpt",
    "imagenet_256_uncond": "imagenet/256x256_diffusion_uncond.pt",
    "imagenet_256_cond": "imagenet/256x256_diffusion.pt",
    "imagenet_256_classifier": "imagenet/256x256_classifier.pt",
    "imagenet_512_classifier": "imagenet/512x512_classifier.pt",
    "imagenet_512_cond": "imagenet/512x512_diffusion.pt",
    "ffhq_256": "ffhq/ffhq_10m.pt",
    "celebahq": "celebahq/celebahq_p2.pt",
    "lsun": "lsun/lsun_bedroom.pt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
    "lsun_bedroom": "f70280ac0e08b8e696f42cb8e948ff1c",
    "ema_lsun_bedroom": "1921fa46b66a3665e450e42f36c2720f",
    "lsun_cat": "bbee0e7c3d7abfb6e2539eaf2fb9987b",
    "ema_lsun_cat": "646f23f4821f2459b8bafc57fd824558",
    "lsun_church": "eb619b8a5ab95ef80f94ce8a5488dae3",
    "ema_lsun_church": "fdc68a23938c2397caba4a260bc2445f",
}


def download(url, local_path, chunk_size=1024):
    if dist.get_rank() == 0:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        if parsed_url.hostname == "drive.google.com" or parsed_url.hostname.endswith(".drive.google.com"):
            gdown.download(url, local_path, quiet=False)
        else:
            os.makedirs(os.path.split(local_path)[0], exist_ok=True)
            with requests.get(url, stream=True) as r:
                total_size = int(r.headers.get("content-length", 0))
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    with open(local_path, "wb") as f:
                        for data in r.iter_content(chunk_size=chunk_size):
                            if data:
                                f.write(data)
                                pbar.update(chunk_size)
    dist.barrier()


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root=None, check=False, prefix='exp'):
    if 'church_outdoor' in name:
        name = name.replace('church_outdoor', 'church')
    assert name in URL_MAP
    # Modify the path when necessary
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.join(prefix, "logs/"))
    root = (
        root
        if root is not None
        else os.path.join(cachedir, "diffusion_models_converted")
    )
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def ckpt_path_adm(name, cfg):
    logger = get_logger('ckpt', cfg)

    ckpt_root = os.path.join(cfg.exp.root, cfg.exp.ckpt_root)
    ckpt = os.path.join(ckpt_root, CKPT_MAP[name])
    if not os.path.exists(ckpt):
        logger.info(URL_MAP[name])
        download(URL_MAP[name], ckpt)
    return ckpt