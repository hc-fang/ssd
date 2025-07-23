

python linear_eval.py --name $1 --dataset_name $2
python semi_eval.py --name $1 --dataset_name $2 --labeled_ratio 0.01
python semi_eval.py --name $1 --dataset_name $2 --labeled_ratio 0.1

