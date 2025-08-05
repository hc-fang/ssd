# Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning




This is the official codebase for *Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning* (ICCV'25) by Hung-Chieh Fang, Hsuan-Tien Lin, Irwin King and Yifei Zhang.


[[paper]()] [[website](https://ssd-uniformity.github.io/)] 

## Installation

```
conda create -n ssd python=3.9
pip install -r requirements.txt
```


## Data preparation

1. Download and preprocess the dataset for federated learning. Run the following script:
    ```
    python split_dataset.py
    ```

    You can customize the dataset generation by modifying the following parameters:
    *  `dataset_name`: Choose from `cifar10`, `cifar100`, `tinyimagenet200`
    *  `client_num`: Set the number of clients.
    *  `alpha`: Controls the level of data heterogeneity using a Dirichlet distribution $\text{Dir}(\boldsymbol{\alpha})$; lower values result in more heterogeneous client data.

2. You can check the images under `data/fed/` to view the label distribution for each client.
3. We have provided processed FL setup files under `data` for better reproducing our experiments. 


## Training

* Run the original FedAlignUniform

    ```
    ./scripts/run_FedAlignUniform.sh $dataset_name $n_clients $join_ratio
    ```

* Run SSD (FedAlignUniform + DSR + PD)

    ```
    ./scripts/run_FedAlignUniformSubspace.sh $dataset_name $n_clients $join_ratio
    ```

* Example usage: 
    ```
    ./scripts/run_FedAlignUniformSubspace.sh cifar100 50 0.2
    ```

## Evaluation

* We evaluate with linear probing and semi-supervised learning (training on 1% and 10% of labeled data):
    ```
    ./scripts/run_eval.sh $run_name $dataset_name
    ```
    * The `$run_name` parameter should be the experiment directory name generated during training. You can find these under your results directory.

* Example usage: 
    ```
    ./scripts/run_eval.sh FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1_KL-weighted10-modefixed-range10_10-normTrue-detachTrue-cifar10-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-07.24-02.56 cifar10
    ```

* We provide the wandb runs that reproduce the main table results: https://wandb.ai/hcfang/SSD/table?nw=nwuserhcfang

## BibTex

```
@inproceedings{ssd_fang2025,
    title={Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning},
    author={Hung-Chieh Fang and Hsuan-Tien Lin and Irwin King and Yifei Zhang},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025},
}
```

## Acknowledgment

The codebase is modified from [FedU2](https://github.com/XeniaLLL/FedU2).
