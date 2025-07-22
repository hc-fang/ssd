import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import wandb

from fedbox.utils.training import set_seed
from models.resnet import ResNet18
from tensorloader import TensorLoader


parser = argparse.ArgumentParser()
parser.add_argument('--torchvision_root', default="./data")
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--name', type=str)
parser.add_argument('--dataset_name', default='cifar10')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--bsz', type=int, default=512)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--projector', action="store_true", default=False)

def main():
    set_seed(1)
    args = parser.parse_args()

    # setup wandb
    wandb.init(
        project='FedSubspace_LinearEval',
        entity='hcfang',
        name=f'{args.name}_p' if args.projector else args.name,
        mode='online'
    )

    if "2d" not in args.name:
        encoder = ResNet18()
        encoder.fc = torch.nn.Identity()
        feat_dim = 512
    else:
        encoder = ResNet18(num_classes=2)
        feat_dim = 2
        
    checkpoint_path = f'outputs/{args.name}/checkpoint_last.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'online_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['online_encoder'])
    elif 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
    elif 'global_net' in checkpoint:
        encoder.load_state_dict({k.removeprefix('backbone.'): v for k, v in checkpoint['global_net'].items() if k.startswith('backbone.')})
    else:
        raise ValueError('invalid checkpoint')

    encoder.to(args.device)
    encoder.eval()
    
    if args.projector:
        projector = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512),
        )
        projector.load_state_dict(checkpoint['projector'])
        projector.to(args.device)
        projector.eval()
    
    # if f'{args.dataset_name}-' not in args.name and f'{args.dataset_name}_' not in args.name:
    #     print("Load wrong dataset")
    #     exit()

    if args.dataset_name == 'cifar10':
        train_set = CIFAR10(args.torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR10(args.torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 10
    else:
        train_set = CIFAR100(args.torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR100(args.torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 100

    with torch.no_grad():
        if args.projector:
            train_z = torch.concat([projector(encoder(x.to(args.device))).cpu() for x, _ in tqdm(DataLoader(train_set, args.bsz), desc='extract train_z')])
            test_z = torch.concat([projector(encoder(x.to(args.device))).cpu() for x, _ in tqdm(DataLoader(test_set, args.bsz), desc='extract test_z')])
        else:
            train_z = torch.concat([encoder(x.to(args.device)).cpu() for x, _ in tqdm(DataLoader(train_set, args.bsz), desc='extract train_z')])
            test_z = torch.concat([encoder(x.to(args.device)).cpu() for x, _ in tqdm(DataLoader(test_set, args.bsz), desc='extract test_z')])
        train_y = torch.tensor(train_set.targets)
        test_y = torch.tensor(test_set.targets)

    encoder.cpu()
    classifier = torch.nn.Linear(feat_dim, class_num)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    classifier.to(args.device)
    
    for epoch in tqdm(range(args.epochs), desc=f'train linear'):
        classifier.train()
        for z, y in TensorLoader((train_z, train_y), batch_size=args.bsz, shuffle=True):
            z, y = z.to(args.device), y.to(args.device)
            output = classifier(z)
            loss = torch.nn.functional.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            classifier.eval()
            with torch.no_grad():
                test_pred = torch.concat([classifier(z.to(args.device)).argmax(dim=1).cpu() for z in TensorLoader(test_z, batch_size=args.bsz)])
                acc = accuracy_score(test_y, test_pred)

            wandb.log({
                "Acc": acc
            })


if __name__ == "__main__":
    main()