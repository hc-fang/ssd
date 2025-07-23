export CONFIG_PATH="configs/FedAlignUniform"
fed=hete # hete, homo
dataset=$1 # cifar10, cifar100, tinyimagenet200
n_clients=$2
join_ratio=$3

python main.py \
    seed=17 \
    fed=$fed \
    params.dataset=$dataset \
    params.n_clients=$n_clients \
    params.join_ratio=$join_ratio