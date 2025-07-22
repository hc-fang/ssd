NAMES=(
	# FedAlignUniformU2-uni1.0-fur0.01-aggprojTrue-BN-bn_projectorTrue-cifar100_al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17_02.26-04.49
    #     FedAlignUniformDecorr-uni1.0-decorr0.01-aggprojTrue-BN-bn_projectorTrue-cifar100_al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17_02.26-04.49
#    FedAlignUniform-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-02.25-18.55
 #   FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1_KL-weighted10-compFalse-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-02.25-18.55
    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1-weighted51-compTrue-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar10-al100.0-num10-jr1.0-rd100-ep5-bsz128-seed17-02.18-12.14
    # FedAlignUniform-uni1.0-dist0.1-aggprojTrue-BN-bn_projectorTrue-cifar10-al100.0-num10-jr1.0-rd100-ep5-bsz128-seed17-02.18-12.21
    # FedAlignUniformDecorr-uni1.0-decorr0.01-aggprojTrue-BN-bn_projectorTrue-cifar10_al100.0-num10-jr1.0-rd100-ep5-bsz128-seed17_02.20-03.44
    # FedAlignUniformSubspace-uni1.0-ss1.0-weighted51-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-cifar10-al100.0-num10-jr1.0-rd100-ep5-bsz128-seed17-02.12-21.25
    # FedAlignUniform-uni1.0-aggprojTrue-cifar10-al100.0-num10-jr1.0-rd100-ep5-bsz128-seed17-02.11-19.21
    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1-weighted10-compFalse-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num50-jr1.0-rd100-ep5-bsz128-seed17-02.20-02.30
    # FedAlignUniform-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num50-jr1.0-rd100-ep5-bsz128-seed17-02.20-02.02
   #  FedAlignUniform-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-02.22-15.43
    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1-weighted10-compFalse-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-02.22-15.48
    # FedAlignUniformDecorr-uni1.0-decorr0.01-aggprojTrue-BN-bn_projectorTrue-cifar10_al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17_02.23-05.25
    # FedAlignUniformU2-uni1.0-fur0.01-aggprojTrue-BN-bn_projectorTrue-cifar10_al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17_02.23-06.12
    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1-weighted10-compFalse-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-02.24-03.10
    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1-weighted10-compFalse-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-02.24-03.14
    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1_MSE-weighted10-compFalse-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-02.24-04.07
    # FedAlignUniformU2-uni1.0-fur0.01-aggprojTrue-BN-bn_projectorTrue-cifar10_al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17_02.24-15.35
    # FedAlignUniformU2-uni1.0-fur0.01-aggprojTrue-BN-bn_projectorTrue-cifar100_al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17_02.24-15.36
    # FedAlignUniform-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-02.18-12.30
    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1-weighted51-compFalse-normTrue-spacefixed-dimnonoverlapping-range2_2-detachTrue-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-02.19-15.06
    # FedAlignUniformDecorr-uni1.0-decorr0.01-aggprojTrue-BN-bn_projectorTrue-cifar100_al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17_02.20-03.43
    # FedAlignUniformU2-uni1.0-fur0.01-aggprojTrue-BN-bn_projectorTrue-cifar100_al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17_02.24-15.36

    # FedAlignUniformSubspace-uni1.0-ss1.0-dist0.1_KL-weighted51-compFalse-normTrue-spacefixed-dimnonoverlapping-range10_10-detachTrue-aggprojTrue-BN-bn_projectorTrue-tinyimagenet200-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-02.27-06.21
    # FedAlignUniform-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-tinyimagenet200-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-02.27-05.55
    # FedAlignUniformDecorr-uni1.0-decorr0.01-aggprojTrue-BN-bn_projectorTrue-tinyimagenet200_al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17_02.27-06.23
    # FedAlignUniformX-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-03.04-01.22
    # FedAlignUniformX-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-03.04-01.22
    # FedAlignUniformX-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-03.04-01.20
    # FedAlignUniformX-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-03.04-00.45
    FedAlignUniformLDAWA-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-03.06-01.36
    # FedAlignUniformLDAWA-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num10-jr1.0-rd100-ep5-bsz128-seed17-03.06-01.31
    FedAlignUniformLDAWA-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar100-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-03.06-01.39
    # FedAlignUniformLDAWA-uni1.0-dist0.0-aggprojTrue-BN-bn_projectorTrue-cifar10-al0.1-num50-jr0.2-rd100-ep5-bsz128-seed17-03.06-01.38
)



dataset=cifar100

export CUDA_VISIBLE_DEVICES=$1

for NAME in "${NAMES[@]}"; do
    python semi_eval.py --name $NAME --dataset_name $dataset --labeled_ratio 0.01
    python semi_eval.py --name $NAME --dataset_name $dataset --labeled_ratio 0.1
done
