import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import json
import copy
import random
import os
import wandb

from methods.fedalignuniform import FedAlignUniformServer, FedAlignUniformClient
from utils.augment import AugPairDataset, GaussianBlur
from datetime import datetime
from models.resnet import ResNet18 
# from mains.subspace import generate_diverse_subspaces

class FedAlignUniform:
    def __init__(
        self,
        task_name: str,
        dataset: str,
        split_file: str,
        torchvision_root: str = "./data",
        device: str = "cuda",
        global_rounds: int = 100,
        local_epochs: int = 5,
        base_lr: float = 0.2,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        temperature: float = 0.07,
        join_ratio: float = 1.0,
        n_clients: int = 10,
        uniform_coeff: float = 1.0,
        # subspace parameters
        subspace_coeff: float = 0.5,
        subspace_weight_dim: int = 50,
        subspace_weight_range: list = [2, 2],
        subspace_mode: str = 'random',
        subspace_norm: bool = True,
        subspace_detach: bool = True,
        
        # distillation
        distill_coeff: float= 0.0,
        distill_method: str = 'KL',
    ):
        self.task_name = task_name
        self.dataset = dataset
        self.split_file = split_file
        self.torchvision_root = torchvision_root
        self.device = torch.device(device)
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.base_lr = base_lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.temperature = temperature
        self.join_ratio = join_ratio
        self.n_clients = n_clients
        self.uniform_coeff = uniform_coeff
        # subspace
        self.subspace_coeff = subspace_coeff
        self.subspace_weight_dim = subspace_weight_dim
        self.subspace_weight_range = subspace_weight_range
        self.subspace_mode = subspace_mode
        self.subspace_norm = subspace_norm
        self.subspace_detach = subspace_detach
        # distillation
        self.distill_coeff = distill_coeff
        self.distill_method = distill_method
        

    def load_data(self):
        with open(self.split_file) as file:
            client_indices = json.load(file)

        if self.dataset == "cifar10":
            train_set = CIFAR10(root=self.torchvision_root, transform=torchvision.transforms.ToTensor())
            test_set = CIFAR10(root=self.torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        elif self.dataset == "cifar100":
            train_set = CIFAR100(root=self.torchvision_root, transform=torchvision.transforms.ToTensor())
            test_set = CIFAR100(root=self.torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        elif self.dataset == 'tinyimagenet200':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),  # Tiny ImageNet has 64x64 images
                torchvision.transforms.ToTensor()
            ])
            train_set = ImageFolder(root=f'{self.torchvision_root}/tiny-imagenet-200/train', transform=transform)
            test_set = CIFAR10(root=self.torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        else:
            raise ValueError("Unsupported dataset")

        return train_set, test_set, client_indices

    def create_model(self):
        encoder = ResNet18()
        projector = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512),
        )
        
        encoder.fc = torch.nn.Identity()

        return encoder, projector

    def run(self):
        train_set, test_set, client_indices = self.load_data()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ])

        train_sets = [Subset(train_set, indices) for indices in client_indices]

        if self.n_clients <= 10:
            aug_train_loaders = [
                DataLoader(AugPairDataset(local_dataset, transforms), batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)
                for local_dataset in train_sets
            ]
        else:
            aug_train_loaders = [
                DataLoader(AugPairDataset(local_dataset, transforms), batch_size=self.batch_size, shuffle=True, num_workers=3)
                for local_dataset in train_sets
            ]

        encoder, projector = self.create_model()
        self.lr = self.base_lr * (self.batch_size / 256)
            

        clients = [
            FedAlignUniformClient(
                id=i,
                encoder=copy.deepcopy(encoder),
                projector=copy.deepcopy(projector),
                aug_train_loader=aug_train_loaders[i],
                train_set=train_sets[i],
                local_epochs=self.local_epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                temperature=self.temperature,
                uniform_coeff=self.uniform_coeff,
                subspace_coeff=self.subspace_coeff,
                subspace_norm=self.subspace_norm,
                subspace_detach=self.subspace_detach,
                distill_coeff=self.distill_coeff,
                distill_method=self.distill_method,
                device=self.device,
            )
            for i in range(self.n_clients)
        ]

        server = FedAlignUniformServer(
            clients=clients,
            encoder=copy.deepcopy(encoder),
            projector=copy.deepcopy(projector),
            test_set=test_set,
            global_rounds=self.global_rounds,
            device=self.device,
            join_ratio=self.join_ratio,
            checkpoint_dir=f"./",

            # subspace
            subspace_weight_dim=self.subspace_weight_dim,
            subspace_weight_range=self.subspace_weight_range,
            subspace_mode=self.subspace_mode,
        )

        server.fit()