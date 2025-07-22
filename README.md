# Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning




This is the official codebase for *Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning* (ICCV'25) by Hung-Chieh Fang, Hsuan-Tien Lin, Irwin King and Yifei Zhang.


[[paper (empty link)]()] [[website](https://ssd-uniformity.github.io/)] 

## Installation

```
conda create -n ssd (fedrepr) python=3.10
pip install -r requirements.txt
```


## Data preparation

1. Download and preprocess the dataset for federated learning

    Run the following script:
    ```
    python split_dataset.py
    ```

    You can customize the dataset generation by modifying the following parameters:
    *  `dataset_name`: Choose from `cifar10`, `cifar100`, `tinyimagenet200`
    *  `client_num`: Set the number of clients.
    *  `alpha`: Controls the level of data heterogeneity using a Dirichlet distribution $\text{Dir}(\boldsymbol{\alpha})$; lower values result in more heterogeneous client data.

2. You can check the images under `data/fed/` to view the label distribution for each client.


## Training

* Run the original FedAlignUniform

```
./scripts/run_FedAlignUniform.sh
```

* Run SSD (FedAlignUniform + DSR + PD)

```
./scripts/run_SSD.sh
```

## Evaluation


## Acknowledgment

The codebase is modified from [FedU2](https://github.com/XeniaLLL/FedU2).