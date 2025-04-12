exp_root=./clean_test/bid/best_config1  # Root and experiment name
ROOT_DIR=$(pwd)
ckpt_root=${ROOT_DIR}/ckpts
save_deg=True
save_ori=True
overwrite=True
smoke_test=1  # Controls the number of batches generated
batch_size=1
use_wandb=False

## Dataset and Model Params ##
dataset=ffhq
config_name=${dataset}256_uncond #dataset
model=${dataset}_uncond
loader=imagenet256_ddrm

w_terminal=50.0  # Options: >=0.0
w_control=ddim
w_score=ddim
combine_fn_name=additive
combine_fn_gamma_t=1.0
algo_init_xT=random
algo_init_control=zero #causal-zero  # Options: zero, random, causal-zero, causal-random
u_lr=0.005
u_lr_scheduler=linear # Options: const, linear

deg_file=bid_gaussian_deblur # Options: superres, non-linear_deblur, bid_motion_deblur, bid_gaussian_deblur
sigma_y=0.005

t_end=1000
t_start=0
difussion_steps=400

M=15  # Num u_t optimization steps
eta=0.7 # For DDIM

RANDOM_PORT=$((8000 + RANDOM % 1000))
echo "Using random port: $RANDOM_PORT"


samples_root=\"M=${M}_steps=${difussion_steps}_lr=${lr}_eta=${eta}_gamma_t=${combine_fn_gamma_t}_w_terminal=${w_terminal}_w_control=${w_control}_w_score=${w_score}_t_end=${t_end}_${model}_u_lr=${u_lr}_scheduler=${u_lr_scheduler}_${deg_file}${factor}\"
python main.py \
    --config-name=$config_name \
    diffusion=ddpm \
    classifier=none \
    algo=ndtm_blind \
    algo.denoise=False \
    algo.M=$M \
    algo.u_lr=$u_lr \
    algo.u_lr_scheduler=$u_lr_scheduler \
    algo.eta=$eta \
    algo.w_terminal=$w_terminal \
    algo.w_score=$w_score \
    algo.w_control=$w_control \
    algo.init_xT=${algo_init_xT} \
    algo.init_control=${algo_init_control} \
    algo.combine_fn.name=${combine_fn_name} \
    algo.combine_fn.gamma_t=${combine_fn_gamma_t} \
    algo.sigma_y=${sigma_y} \
    deg=${deg_file} \
    deg.lr=$lr \
    dataset.subset=100 \
    loader=$loader \
    loader.batch_size=$batch_size \
    loader.num_workers=2 \
    dist.num_processes_per_node=1 \
    dist.port=$RANDOM_PORT \
    exp.t_start=0.0 \
    exp.t_end=${t_end} \
    exp.num_steps=${difussion_steps} \
    exp.seed=0 \
    exp.stride=ddpm_uniform \
    exp.root=$exp_root \
    exp.name=\"M=${M}_steps=${difussion_steps}_eta=${eta}_gamma_t=${combine_fn_gamma_t}_w_terminal=${w_terminal}_w_control=${w_control}_w_score=${w_score}_t_end=${t_end}_u_lr=${u_lr}_scheduler=${u_lr_scheduler}\" \
    exp.ckpt_root=$ckpt_root \
    exp.samples_root=$samples_root \
    exp.overwrite=True \
    exp.use_wandb=$use_wandb \
    exp.save_ori=$save_ori \
    exp.save_deg=$save_deg \
    exp.smoke_test=$smoke_test

echo "Job completed on $(date)"
