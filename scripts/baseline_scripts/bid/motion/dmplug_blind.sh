exp_root=./clean_test/bid_motion/dmplug
samples_root=gaussian_samples/
ROOT_DIR=$(pwd)
ckpt_root=${ROOT_DIR}/ckpts
save_deg=True
save_ori=True
overwrite=True
smoke_test=1 # Controls the number of batches generated
batch_size=1

# Fixed Parameters

dataset=ffhq
config_name="${dataset}256_uncond"
model="${dataset}_uncond"
loader=imagenet256_ddrm
deg_file=bid_motion_deblur

lrk=0.1
lr=0.01
num_inference_steps=3
eta=0.0  # For DDIM
t_start=0.0
t_end=1000.0
epochs=10000

python main.py \
    --config-name=$config_name \
    diffusion=ddpm \
    classifier=none \
    algo=dmplug_blind \
    algo.eta=$eta \
    algo.num_inference_steps=$num_inference_steps \
    algo.lr=$lr \
    algo.lrk=$lrk \
    algo.epochs=$epochs \
    deg=${deg_file} \
    dataset.subset=100 \
    loader=$loader \
    loader.batch_size=$batch_size \
    loader.num_workers=2 \
    dist.num_processes_per_node=1 \
    dist.port=8799 \
    exp.name=debug \
    exp.use_wandb=False \
    exp.t_start=${t_start} \
    exp.t_end=${t_end} \
    exp.seed=0 \
    exp.stride=ddpm_uniform \
    exp.root=$exp_root \
    exp.name="test" \
    exp.ckpt_root=$ckpt_root \
    exp.samples_root=$samples_root \
    exp.overwrite=True \
    exp.save_ori=$save_ori \
    exp.save_deg=$save_deg \
    exp.smoke_test=$smoke_test
  # done
# done
