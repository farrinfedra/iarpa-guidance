exp_root=/home/pandeyk1/coc_results/ffhq/sr4/dtm/sota1k/ #the root and the exp_name.
ROOT_DIR=$(pwd)
ckpt_root=${ROOT_DIR}/ckpts
save_deg=True
save_ori=True
overwrite=True
smoke_test=100 # Controls the number of batches generated
batch_size=10

## Dataset and Model Params ##
config_name=ffhq256_uncond #dataset
model=ffhq_uncond
loader=imagenet256_ddrm  # imagenet256_ddrmpp for imagenet


etas=(0.7)
gammas=(1.0)
t_ends=(400)

for t_end in "${t_ends[@]}"; do
for combine_fn_gamma_t in "${gammas[@]}"; do
for eta in "${etas[@]}"; do

echo "Running script for: t-end $t_end and gamma_t: $combine_fn_gamma_t"

### Tunable params ###
algo=ndtm
w_terminal=50.0  # Weight for terminal loss
w_control=ddim # Options: zero, ones, ddim, ddpm
w_score=ddim
algo_init_xT=guided  # Options: guided, random
algo_init_control=zero #causal-zero  # Options: zero, random, causal-zero, causal-random
combine_fn_name=additive  # Options: additive, elementwise
# combine_fn_gamma_t=2.0
u_lr=0.01  # LR for optimizing u_t
u_lr_scheduler=linear # Options: const, linear

### Degradation parameters ###
deg_file=superres # Options: superres, non-linear_deblur, bid_motion_deblur, bid_gaussian_deblur
factor=4  # NOTE: Keep only if deg_file=superres
sigma_y=0.005

### Fixed params ###
difusion_steps=50
M=5  # Num u_t optimization steps
# eta=0.7  # DDIM stochasticity parameter (0 for deterministic sampling)
t_start=0
# t_end=400

samples_root=\"best_lpips_${algo}_${model}_u_lr=${u_lr}_init_xT=${algo_init_xT}_M=${M}_t_start=${t_start}_t_end=${t_end}_gamma_t=${combine_fn_gamma_t}_eta=${eta}_w_terminal=${w_terminal}_w_control=${w_control}_w_score=${w_score}_${deg_file}${factor}\"

RANDOM_PORT=$((8000 + RANDOM % 1000))
echo "Using random port: $RANDOM_PORT"

# Main Python script execution
python main.py \
    --config-name=$config_name \
    diffusion=ddpm \
    classifier=none \
    algo=$algo \
    algo.denoise=False \
    algo.M=$M \
    algo.sigma_y=$sigma_y \
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
    deg=${deg_file} \
    deg.factor=${factor} \
    loader=$loader \
    loader.batch_size=$batch_size \
    loader.num_workers=2 \
    dist.num_processes_per_node=1 \
    dist.port=$RANDOM_PORT \
    exp.t_start=$t_start \
    exp.t_end=$t_end \
    exp.num_steps=${difusion_steps} \
    exp.seed=0 \
    exp.stride=ddpm_uniform \
    exp.root=$exp_root \
    exp.name=samples \
    exp.ckpt_root=$ckpt_root \
    exp.samples_root=$samples_root \
    exp.overwrite=True \
    exp.use_wandb=False \
    exp.save_ori=$save_ori \
    exp.save_deg=$save_deg \
    exp.smoke_test=$smoke_test

echo "Job completed on $(date)"

done
done
done