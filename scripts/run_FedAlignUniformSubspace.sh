export CONFIG_PATH="configs/FedAlignUniformSubspace"
export CUDA_VISIBLE_DEVICES=$1
subspace_coeff=1.0
distill_coeff=0.1
subspace_weight_dim=51
fed=hete
n_clients=10
dataset=cifar10 # cifar100
join_ratio=1.0
distill_method='KL'

python main.py \
    seed=$2 \
    params.subspace_coeff=$subspace_coeff \
    params.distill_coeff=$distill_coeff \
    params.subspace_weight_dim=$subspace_weight_dim \
    fed=$fed \
    params.dataset=$dataset \
    params.n_clients=$n_clients \
    params.join_ratio=$join_ratio \
    params.distill_method=$distill_method
