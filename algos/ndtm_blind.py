import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import wandb
from models.classifier_guidance_model import ClassifierGuidanceModel
from utils import register_module
from utils.combine_fn import *
from utils.degredations import build_degredation_model, get_degreadation_image
from utils.functions import postprocess

from .ddim import DDIM


@register_module(category="algo", name="ndtm_blind")
class NDTM_BLIND(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):

        self.model = model
        self.diffusion = model.diffusion
        self.cfg = cfg
        self.eta = cfg.algo.eta
        self.sdedit = cfg.algo.sdedit
        self.snrs = []

        self.M = self.cfg.algo.M  # Number of optimization steps
        self.gamma_t = self.cfg.algo.combine_fn.get("gamma_t", None)
        self.u_lr = self.cfg.algo.u_lr

        self.H = build_degredation_model(cfg)
        self.F = self._get_f()

    def _get_f(self):
        combine_fn = self.cfg.algo.combine_fn
        if combine_fn.name == "additive":
            return Additive(gamma_t=self.gamma_t)

    def _get_score_weight(self, scheme, t, s, **kwargs):

        allowed_schemes = {
            "ddpm",
            "ddim",
            "zero",
            "ones",
            "reddiff_linear",
            "reddiff_sqrt",
            "reddiff_square",
            "reddiff_log",
            "reddiff_trunc_linear",
            "reddiff_power2over3",
            "reddiff_const",
        }
        assert scheme in allowed_schemes, f"Unknown scheme: {scheme}"

        alpha_t = self.diffusion.alpha(t)  # this is 0
        alpha_s = self.diffusion.alpha(s)
        beta_t = self.diffusion.beta(t)
        alpha_t_im = 1 - beta_t

        if scheme == "zero":
            return torch.tensor([0.0], device=alpha_s.device)
        elif scheme == "ones":
            return torch.tensor([1.0], device=alpha_s.device)
        elif scheme == "ddpm":
            return (beta_t**2) / (alpha_t_im * (1 - alpha_t))
        elif scheme == "ddim":
            c1 = (
                (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
            ).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1**2).sqrt()
            c2_ = ((alpha_s / alpha_t) * (1 - alpha_t)).sqrt()
            return (c2 - c2_) ** 2

        elif "reddiff" in scheme:
            snr = (
                alpha_t[0].sqrt() / (1 - alpha_t[0]).sqrt()
            )  # 1d torch tensor #noise to signal ratio

            self.snrs.append(snr.item())
            if "linear" in scheme:
                snr = snr
            elif "sqrt" in scheme:
                snr = torch.sqrt(snr)
            elif "square" in scheme:
                snr = torch.square(snr)
            elif "log" in scheme:
                kwargs["logger"].info(f"Using Score weight: {snr}")
                snr = torch.log(snr + 1.0)
            elif "trunc_linear" in scheme:
                snr = torch.clip(snr, max=1.0)
            elif "power2over3" in scheme:
                snr = torch.pow(snr, 2 / 3)
            elif "const" in scheme:
                snr = torch.pow(snr, 0.0)
            return snr

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def _get_control_weight(self, scheme, t, s):
        assert scheme in ["ddpm", "ddim", "zero", "ones"]

        alpha_t = self.diffusion.alpha(t)  # this is 0
        alpha_s = self.diffusion.alpha(s)
        beta_t = self.diffusion.beta(t)
        alpha_t_im = 1 - beta_t

        if scheme == "zero":
            return torch.tensor([0.0], device=alpha_s.device)
        elif scheme == "ones":
            return torch.tensor([1.0], device=alpha_s.device)
        elif scheme == "ddpm":
            return 1 / alpha_t_im
        else:
            return alpha_t / alpha_s

    def get_learning_rate(self, base_lr, current_step, total_steps):
        if self.cfg.algo.u_lr_scheduler == "linear":
            return base_lr * (1.0 - current_step / total_steps)
        else:  # const
            return base_lr

    def sample(self, x, y, ts, **kwargs):
        x_orig = x.clone()
        x = self.initialize(x, y, ts, **kwargs)
        y_0 = kwargs["y_0"]
        bs = x.size(0)
        xt = x

        ss = [-1] + list(ts[:-1])
        xt_s = [xt.cpu()]
        x0_s = []
        uts = []

        kernel_size = self.cfg.deg.kernel_size
        trainable_kernel = torch.randn(
            (bs, kernel_size * kernel_size),
            device=x.device,
            requires_grad=True,
        )

        kernel_optimizer = torch.optim.Adam([trainable_kernel], lr=0.01)
        kwargs["logger"].info(f"xt shape: {xt.shape}\ny_0 shape: {y_0.shape}")
        u_t = torch.zeros_like(xt)
        for i, (ti, si) in enumerate(zip(reversed(ts), reversed(ss))):

            t = torch.ones(bs).to(x.device).long() * ti
            s = torch.ones(bs).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = (
                (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
            ).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1**2).sqrt()

            # Initialize control and the optimizer
            u_t = self.initialize_ut(u_t, i)
            ut_clone = u_t.clone().detach()
            ut_clone.requires_grad = True
            current_lr = self.get_learning_rate(self.u_lr, i, len(ts))
            optimizer = torch.optim.Adam([ut_clone], lr=current_lr)

            # Loss weightings
            w_terminal = self.cfg.algo.w_terminal
            w_score = self._get_score_weight(self.cfg.algo.w_score, t, s, **kwargs)
            w_control = self._get_control_weight(self.cfg.algo.w_control, t, s)

            ####################################################
            ############## Control Optimization ################
            ####################################################

            et = self.model(xt, y, t).detach()
            for _ in range(self.M):
                # Guided state vector
                cxt = self.F(xt, ut_clone, **kwargs)

                # guided noise estimates
                et_control = self.model(cxt, y, t)

                # Tweedie's estimate from the guided state vector
                x0_pred = self.diffusion.predict_x_from_eps(cxt, et_control, t)
                x0_pred = torch.clamp(x0_pred, -1, 1)

                score_diff = ((et - et_control) ** 2).view(bs, -1).sum(dim=1)
                c_score = w_score * score_diff

                # Control loss
                control_loss = (
                    ((self.F(xt, ut_clone, **kwargs) - xt) ** 2).view(bs, -1).sum(dim=1)
                )
                c_control = w_control * control_loss * (self.gamma_t**2)

                # Terminal Cost
                kernel_output = F.softmax(trainable_kernel, dim=1)
                out_k = kernel_output.view(bs, 1, kernel_size, kernel_size)
                c_terminal = (
                    ((y_0 - self.H.H(x0_pred, out_k)) ** 2).view(bs, -1).sum(dim=1)
                )
                c_terminal = w_terminal * c_terminal
                c_t = c_score + c_control + c_terminal

                if kwargs["dist"].get_rank() == 0:
                    print(
                        f"Diffusion step: {ti} Terminal Loss: {c_terminal.mean().item()} "
                        f"Control loss: {c_control.mean().item()} Score loss: {c_score.mean().item()}"
                    )

                optimizer.zero_grad()
                kernel_optimizer.zero_grad()
                c_t.sum().backward(retain_graph=True)
                optimizer.step()
                kernel_optimizer.step()

                if kwargs["dist"].get_rank() == 0:
                    if self.cfg.exp.use_wandb:
                        wandb.log(
                            {
                                "c_score": c_score.mean().item(),
                                "c_control": c_control.mean().item(),
                                "c_terminal": c_terminal.mean().item(),
                            }
                        )

            ###########################################
            ############## DDIM update ################
            ###########################################
            with torch.no_grad():

                u_t = ut_clone.detach()
                cxt = self.F(xt, u_t, **kwargs)
                et_control = self.model(cxt, y, t)
                x0_pred = self.diffusion.predict_x_from_eps(cxt, et_control, t)
                x0_pred = torch.clamp(x0_pred, -1, 1)
                xt = (
                    alpha_s.sqrt() * x0_pred
                    + c1 * torch.randn_like(xt)
                    + c2 * et_control
                )
                uts.append(u_t.cpu())

            xt_s.append(xt.cpu())
            x0_s.append(x0_pred.cpu())

            # Plot to wandb
            if kwargs["dist"].get_rank() == 0:
                if self.cfg.exp.use_wandb:
                    xo = postprocess(x0_pred).cpu()

                    deg_image = postprocess(
                        get_degreadation_image(y_0, self.H, self.cfg)
                    ).cpu()

                    if self.cfg.exp.use_wandb:
                        comparison = torch.cat(
                            [x_orig.cpu()[0], deg_image[0], xo[0]], dim=2
                        )
                        comparison_image = comparison.permute(1, 2, 0).detach().numpy()

                        wandb.log(
                            {
                                f"Comparison": wandb.Image(
                                    comparison_image,
                                    caption=f"Original | psudo-inverse: {self.cfg.deg.name} | x_0pred",
                                )
                            }
                        )

        return list(reversed(xt_s)), list(reversed(x0_s))

    def initialize_ut(self, ut, i):
        init_control = self.cfg.algo.init_control

        if init_control == "zero":  # constant zero
            return torch.zeros_like(ut)
        elif init_control == "random":  # constant random
            return torch.randn_like(ut)

        elif "causal" in init_control:
            if "zero" in init_control and i == 0:  # causal_zero
                return torch.zeros_like(ut)
            elif "random" in init_control and i == 0:  # causal_random
                return torch.randn_like(ut)

            else:
                return ut

    def initialize(self, x, y, ts, **kwargs):
        """
        random: Initialization with x_T ~ N(0, 1)
        guided: Initialization with x_T ~ DDPM(H^(y_0)) - Only for Linear IP
        """
        init_scheme = self.cfg.algo.init_xT
        if self.cfg.deg.name in ["deblur_nl", "bid"] and init_scheme == "guided":
            kwargs["logger"].info(
                "Guided initialization not supported for non-linear degradations. Switching to random initialization."
            )
            init_scheme = self.cfg.algo.init_xT = "random"

        if init_scheme == "random":
            return super().initialize(x, y, ts, **kwargs)

        elif init_scheme == "guided":
            y_0 = kwargs["y_0"]
            H = self.H
            n = x.size(0)
            x_0 = H.H_pinv(y_0).view(*x.size()).detach()
            ti = ts[-1]
            t = torch.ones(n).to(x.device).long() * ti
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            return alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)
