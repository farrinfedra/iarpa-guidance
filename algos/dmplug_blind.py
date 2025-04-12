import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from utils import register_module
from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import build_degredation_model
from .ddim import DDIM

from utils.ddim_huggingface import *


@register_module(category="algo", name="dmplug_blind")
class DMPlug_Blind(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.scheduler = DDIMScheduler()
        self.diffusion = model.diffusion
        self.cfg = cfg
        self.eta = cfg.algo.eta
        self.sdedit = cfg.algo.sdedit
        self.H = build_degredation_model(cfg)
        self.scheduler.set_timesteps(self.cfg.algo.num_inference_steps)

    def sample(self, x, y, ts, **kwargs):

        y_0 = kwargs["y_0"]

        bs = x.shape[0]
        w = x.shape[2]
        h = x.shape[3]
        dtype = torch.float32
        device = x.device

        Z = torch.randn(
            (bs, 3, w, h),
            device=x.device,
            dtype=dtype,
            requires_grad=True,
        )

        criterion = torch.nn.MSELoss().to(device)
        kernel_size = self.cfg.deg.kernel_size
        trainable_kernel = torch.randn(
            (1, kernel_size * kernel_size),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        params_group1 = {"params": Z, "lr": self.cfg.algo.lr}
        params_group2 = {"params": trainable_kernel, "lr": self.cfg.algo.lrk}
        optimizer = torch.optim.Adam([params_group1, params_group2])

        epochs = (
            self.cfg.algo.epochs
        )  # SR, inpainting: 5,000, nonlinear deblurring: 10,000
        losses = []
        ss = [-1] + list(ts[:-1])
        xt_s = []
        for iterator in range(epochs):
            self.model.model.eval()
            optimizer.zero_grad()
            for i, tt in enumerate(self.scheduler.timesteps):
                t = (torch.ones(1) * tt).to(x.device)
                if i == 0:
                    noise_pred = self.model(Z, y, t)
                else:
                    noise_pred = self.model(xt, y, t)

                if i == 0:
                    xt = self.scheduler.step(
                        noise_pred,
                        tt,
                        Z,
                        return_dict=True,
                        use_clipped_model_output=True,
                        eta=self.cfg.algo.eta,
                    ).prev_sample
                else:
                    xt = self.scheduler.step(
                        noise_pred,
                        tt,
                        xt,
                        return_dict=True,
                        use_clipped_model_output=True,
                        eta=self.cfg.algo.eta,
                    ).prev_sample

            xt = torch.clamp(xt, -1, 1)
            xt_s.append(xt.detach().cpu())
            kernel_output = F.softmax(trainable_kernel, dim=1)
            out_k = kernel_output.view(bs, 1, kernel_size, kernel_size)
            loss = criterion(self.H.H(xt, out_k), y_0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f"Iteration: {iterator}, Loss: {losses[-1]}")

        return list(reversed(xt_s)), None

    def initialize(self, x, y, ts):
        return torch.randn_like(x)
