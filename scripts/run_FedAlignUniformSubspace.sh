export CONFIG_PATH="configs/FedAlignUniformSubspace"

# setup 
fed=hete
dataset=$1 # cifar10, cifar100
n_clients=$2
join_ratio=$3

# training 
subspace_coeff=1.0 
distill_coeff=0.1
subspace_weight_dim=10 # subspace dimension per client (n_dimensions // n_clients); With n_dimensions=512: 10 for n_clients=50, 51 for n_clients=10
distill_method='KL'


python main.py \
    seed=17 \
    params.subspace_coeff=$subspace_coeff \
    params.distill_coeff=$distill_coeff \
    params.subspace_weight_dim=$subspace_weight_dim \
    fed=$fed \
    params.dataset=$dataset \
    params.n_clients=$n_clients \
    params.join_ratio=$join_ratio \
    params.distill_method=$distill_method
