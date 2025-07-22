import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import wandb

from fedbox.utils.training import set_seed
from models.resnet import ResNet18
from tensorloader import TensorLoader
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--torchvision_root', default="./data")
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--name', type=str)
parser.add_argument('--dataset_name', default='cifar10')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--bsz', type=int, default=512)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--projector', action="store_true", default=False)
parser.add_argument('--labeled_ratio', type=float, default=0.01)

def main():
    set_seed(17)
    args = parser.parse_args()

    # setup wandb
    wandb.init(
        project='FedSubspace_Semi',
        entity='hcfang',
        name=f'{args.name}_ratio{args.labeled_ratio}_p' if args.projector else f'{args.name}_ratio{args.labeled_ratio}',
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
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.64, 1.0), ratio=(1.0, 1.0), antialias=True),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    
    # if f'{args.dataset_name}-' not in args.name and f'{args.dataset_name}_' not in args.name:
    #     print("Load wrong dataset")
    #     exit()

    if args.dataset_name == 'cifar10':
        train_set = CIFAR10(args.torchvision_root, transform=transform)
        test_set = CIFAR10(args.torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 10
    else:
        train_set = CIFAR100(args.torchvision_root, transform=transform)
        test_set = CIFAR100(args.torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 100
    
    labeled_indices, _ = train_test_split(list(range(len(train_set))), train_size=args.labeled_ratio, stratify=train_set.targets)
    labeled_set = Subset(train_set, labeled_indices)
    train_loader = DataLoader(labeled_set, batch_size=128, shuffle=True, num_workers=8, persistent_workers=True)
    classifier = torch.nn.Linear(512, class_num)

    encoder.to(args.device)
    classifier.to(args.device)
    encoder.train()
    classifier.train()
    optimizer = torch.optim.SGD([
        {'params': encoder.parameters()},
        {'params': classifier.parameters(), 'lr': 0.3},
    ], lr=1e-3, momentum=0.9)

    def run_test() -> float:
        encoder.eval()
        classifier.eval()
        y_list = []
        pred_list = []
        with torch.no_grad():
            for x, y in DataLoader(test_set, batch_size=128):
                x, y = x.to(args.device), y.to(args.device)
                pred = classifier(encoder(x)).argmax(dim=1)
                y_list.append(y.cpu())
                pred_list.append(pred.cpu())
        acc = accuracy_score(torch.concat(y_list), torch.concat(pred_list))
        return float(acc)

    best_acc = 0.0
    for epoch in tqdm(range(200)):
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(args.device), y.to(args.device)
            output = classifier(encoder(x))
            loss = torch.nn.functional.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            acc = run_test()
            best_acc = max(best_acc, acc)
            tqdm.write(f'({args.labeled_ratio} labeled) epoch {epoch}, semi eval acc: {acc}')
            encoder.train()
            classifier.train()

            wandb.log({"acc": acc})
            
    print(f'({args.labeled_ratio} labeled) best semi eval acc: {best_acc}')


if __name__ == "__main__":
    main()