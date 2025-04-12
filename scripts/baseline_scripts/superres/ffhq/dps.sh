
ROOT_DIR=$(pwd)
ckpt_root=${ROOT_DIR}/ckpts
save_deg=True
save_ori=True
overwrite=True
smoke_test=10 #0  # Controls the number of batches generated
batch_size=5 #8

## Dataset and Model Params ##
dataset_name=ffhq
config_name="${dataset_name}256_uncond" #dataset
model="${dataset_name}_uncond"
loader=imagenet256_ddrm  # imagenet256_ddrmpp for imagenet

### Tunable params ###
#task specific
deg_file=superres # Options: superres, non-linear_deblur, bid_motion_deblur, bid_gaussian_deblur
factor=8  # NOTE: Keep only if deg_file=superres
sigma_y=0.005

#algo specific
algo_name=dps
eta=0.5  # DDIM stochasticity parameter (0 for deterministic sampling)
t_start=0
t_end=1000
difussion_steps=1000
grad_term_weight=1.0

exp_root="./exps/${algo_name}/${dataset_name}"  # Root and experiment name
samples_root=\"task=${deg_file}_eta=${eta}_grad-term-weight=${grad_term_weight}_t-end=${t_end}_steps=${difussion_steps}\"
exp_name=\"exp\"

RANDOM_PORT=$((8000 + RANDOM % 1000))
echo "Using random port: $RANDOM_PORT"


python main.py \
    --config-name=$config_name \
    diffusion=ddpm \
    classifier=none \
    algo=${algo_name} \
    algo.eta=$eta \
    algo.sigma_y=${sigma_y} \
    algo.grad_term_weight=${grad_term_weight} \
    deg=${deg_file} \
    deg.factor=${factor} \
    dataset.subset=1000 \
    loader=$loader \
    loader.batch_size=$batch_size \
    loader.num_workers=2 \
    dist.num_processes_per_node=1 \
    dist.port=$RANDOM_PORT \
    exp.t_start=${t_start} \
    exp.t_end=${t_end} \
    exp.num_steps=${difussion_steps} \
    exp.seed=0 \
    exp.stride=ddpm_uniform \
    exp.root=$exp_root \
    exp.name=$exp_name \
    exp.ckpt_root=$ckpt_root \
    exp.samples_root=$samples_root \
    exp.overwrite=True \
    exp.save_ori=$save_ori \
    exp.save_deg=$save_deg \
    exp.smoke_test=$smoke_test

echo "Job completed on $(date)"