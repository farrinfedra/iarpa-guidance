<h1 align="center">Variational Control for Guidance in Diffusion Models</h1>
<p align="center" style="font-size: 1.5em;"><strong>ICML&nbsp;2025</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.03686">
    <img src="https://img.shields.io/badge/arXiv-2502.03686-b31b1b.svg" alt="arXiv">
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2502.03686">
    <img src="https://img.shields.io/badge/Project_Page-purple" alt="Project Page">
  </a>
</p>

<div align="center">
  <a href="https://kpandey008.github.io/" target="_blank">Kushagra Pandey *</a>
  &emsp;<b>·</b>&emsp;
  <a href="https://farrinsofian.com" target="_blank">Farrin Marouf Sofian *</a>
  &emsp;<b>·</b>&emsp;
  <a href="https://hci.iwr.uni-heidelberg.de/vislearn/people/felix-draxler/" target="_blank">Felix Draxler</a>
  &emsp;<b>·</b>&emsp;
  <a href="https://karaletsos.com/" target="_blank">Theofanis Karaletsos</a>
  &emsp;<b>·</b>&emsp;
  <a href="https://www.stephanmandt.com/" target="_blank">Stephan Mandt</a>
</div>

<br>

## 🧠 Overview

<p align="center">
  <img src="assets/method.gif" width="85%">
</p>

We formulate guidance in diffusion models as a **variational control** problem. Intuitively, at each diffusion sampling step, we infer controls (via an optimization routine) which guide the noisy state towards a direction which best satisfies a terminal cost subject to a regularization penalty.


## 🔧 Installation

This repo builds on top of the official [RED-Diff](https://github.com/NVlabs/RED-diff) implementation. Clone the repository and navigate to the project directory:

```bash
git clone git@github.com:czi-ai/oc-guidance.git
cd oc-guidance
```

We provide both a Conda environment file (environment.yml) and a requirements.txt. You can use either of the following methods to set up the environment:

```bash
conda env create -f environment.yml
conda activate oc-guidance

# OR
conda create --name <env_name> python=3.11
conda activate <env_name>
pip install -r requirements.txt
```

## 📦 Pretrained Checkpoints and Dataset

In this work, we use:

- **Unconditional ImageNet checkpoints** from the [ADM](https://github.com/openai/guided-diffusion) repository.
- **Pretrained unconditional FFHQ checkpoint** from the [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) repository.

> 📁 **For convenience**, we recommend placing the checkpoints in a folder named `ckpts/` under the corresponding dataset directory.  
> For example:
>
> ```
> ckpts/
> ├── imagenet/
> │       └── model.pt
> └── ffhq/
>         └── model.pt
> ```

Make sure to update the `ckpt_root` argument in config and bash files accordingly when running inference.

### 📁 Dataset

- For **FFHQ**, we selected the **first 1000 validation images** for all tasks except Blind Image Deblurring (BID), where we used only **100 images**.
- For **ImageNet**, we randomly selected **1000 images**.

We provide text files listing the selected image from Imagenet under the `misc/` directory:
- `sr3_top1k.txt`: used for all tasks **except** BID.
- `sr3_top100.txt`: used **only** for the BID task.

---

## ⚙️ Config Management

Configuration management is handled using [Hydra](https://hydra.cc/).  
All configuration files are located in the [`_configs/`](./_configs) directory, with algorithm-specific configurations.


## 🧪 Inference

We provide the **exact scripts used in our experiments**—both for our method and for the baselines—under the [`scripts/`](./scripts) directory. These scripts are organized by task (e.g., super-resolution, deblurring, inpainting) to ensure easy reproducibility.

Below is a minimal example command for running **non-linear deblurring** with our method on ImageNet:

```bash
python main.py \
    --config-name=imagenet256_uncond \
    diffusion=ddpm \
    classifier=none \
    algo=ndtm \
    algo.denoise=False \
    algo.M=2 \
    algo.sigma_y=0.005 \
    algo.u_lr=0.01 \
    algo.u_lr_scheduler=linear \
    algo.eta=0.1 \
    algo.w_terminal=50.0 \
    algo.w_score=ddim \
    algo.w_control=ddim \
    algo.init_xT=guided \
    algo.init_control=zero \
    algo.combine_fn.name=additive \
    algo.combine_fn.gamma_t=4.0 \
    deg=non-linear_deblur \
    loader=imagenet256_ddrm \
    loader.batch_size=1 \
    loader.num_workers=2 \
    dist.num_processes_per_node=1 \
    dist.port=8051 \
    exp.t_start=0 \
    exp.t_end=600 \
    exp.num_steps=50 \
    exp.seed=0 \
    exp.stride=ddpm_uniform \
    exp.root=/path/to/experiment/root \
    exp.name=samples \
    exp.ckpt_root=/path/to/pretrained/checkpoints \
    exp.samples_root=/path/to/save/samples \
    exp.overwrite=True \
    exp.use_wandb=False \
    exp.save_ori=True \
    exp.save_deg=True \
    exp.smoke_test=1
```
To simplify usage, you can instead run the provided shell script directly:

```
sh scripts/method_scripts/non_linear_deblur/imagenet/ndtm.sh
```
## 📚 Citation

```bibtex
@misc{pandey2025variationalcontrolguidancediffusion,
  title={Variational Control for Guidance in Diffusion Models}, 
  author={Kushagra Pandey and Farrin Marouf Sofian and Felix Draxler and Theofanis Karaletsos and Stephan Mandt},
  year={2025},
  eprint={2502.03686},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.03686}, 
}
```
## ✉️ Questions or Contact

For questions, clarifications, or collaborations, feel free to reach out to the authors:

- 📫 **Kushagra Pandey** – [pandeyk1@uci.edu](mailto:pandeyk1@uci.edu) – [kpandey008.github.io](https://kpandey008.github.io)
- 📫 **Farrin Marouf Sofian** – [fmaroufs@uci.edu](mailto:fmaroufs@uci.edu) – [farrinsofian.com](https://farrinsofian.com)  

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
