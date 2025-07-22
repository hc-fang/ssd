export CONFIG_PATH="configs/FedAlignUniform"
export CUDA_VISIBLE_DEVICES=$1
fed=hete
dataset=cifar10
n_clients=10
join_ratio=1.0

python main.py \
    seed=$2 \
    fed=$fed \
    params.dataset=$dataset \
    params.n_clients=$n_clients \
    params.join_ratio=$join_ratio