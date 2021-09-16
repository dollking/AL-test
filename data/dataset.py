from PIL import Image

from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10


class Dataset_CIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super(Dataset_CIFAR10, self).__init__(root, download=download, train=train, transform=transform,
                                              target_transform=target_transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        origin = Image.fromarray(img)

        if self.transform is not None:
            trans = self.transform(origin)
        else:
            trans = origin

        origin = transforms.ToTensor()(origin)
        origin = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(origin)

        return {'origin': origin, 'trans': trans, 'target': target}


class Dataset_CIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super(Dataset_CIFAR100, self).__init__(root, download=download, train=train, transform=transform,
                                               target_transform=target_transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        origin = Image.fromarray(img)

        if self.transform is not None:
            trans = self.transform(origin)
        else:
            trans = origin

        origin = transforms.ToTensor()(origin)
        origin = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(origin)

        return {'origin': origin, 'trans': trans, 'target': target}
