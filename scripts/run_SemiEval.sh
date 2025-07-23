dataset_name=cifar10

python semi_eval.py --name $1 --dataset_name $dataset_name --labeled_ratio 0.01
python semi_eval.py --name $1 --dataset_name $dataset_name --labeled_ratio 0.1

