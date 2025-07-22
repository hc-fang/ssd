import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageFilter



class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        # img = self.pil_to_tensor(img).unsqueeze(0)
        img.unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class AugPairDataset(Dataset):
    def __init__(self, dataset, transform, raw=False):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.raw = raw

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x1, x2 = self.transform(x), self.transform(x)
        if self.raw:
            return x1, x2, x, y
        else:
            return x1, x2

    def __len__(self) -> int:
        return len(self.dataset)



class AugPairRotDataset(Dataset):
    def __init__(self, dataset, transform):
        super(AugPairRotDataset).__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        n = random.random()
        angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
        # angle = 0 if n <= 0.125 else 1 if n <= 0.25 else 2 if n <= 0.375 else 3 if n <= 0.5 else 4 if n <= 0.625 else 5 if n <= 0.75 else 6 if n <= 0.875 else 7

        x1, x2 = self.transform(x), self.transform(x)
        x3 = image_rot(self.transform(x), 90 * angle)
        return x1, x2, x3, angle

    def __len__(self) -> int:
        return len(self.dataset)

class AugTripleDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        x1, x2, x3 = self.transform(x), self.transform(x), self.transform(x)
        return x, x1, x2, x3

    def __len__(self) -> int:
        return len(self.dataset)


def image_rot(image, angle):
    image = TF.rotate(image, angle)
    return image


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class AugOrchestraDataset(Dataset):
    def __init__(self, dataset, is_sup):
        super(AugOrchestraDataset, self).__init__()
        image_size = 32
        self.transform = Compose([
            RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(p=0.5),
            RandomApply([ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_prime = Compose([
            RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(p=0.5),
            RandomApply([ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            Solarization(p=0.2),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = dataset

        self.mode = is_sup

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        n = random.random()
        angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
        # angle = 0 if n <= 0.125 else 1 if n <= 0.25 else 2 if n <= 0.375 else 3 if n <= 0.5 else 4 if n <= 0.625 else 5 if n <= 0.75 else 6 if n <= 0.875 else 7
        if (self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform_prime(x)
            x3 = image_rot(self.transform(x), 90 * angle)
            # x3 = image_rot(self.transform(x), 45 * angle)
            return x1, [x2, x3, angle]

    def __len__(self) -> int:
        return len(self.dataset)

