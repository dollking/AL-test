from PIL import Image
from torchvision.datasets import CIFAR100, CIFAR10


class Dataset_CIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super(Dataset_CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        origin = Image.fromarray(img)

        if self.transform is not None:
            trans = self.transform(origin)
        else:
            trans = origin

        return {'origin': origin, 'trans': trans}


class Dataset_CIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super(Dataset_CIFAR100, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        origin = Image.fromarray(img)

        if self.transform is not None:
            trans = self.transform(origin)
        else:
            trans = origin

        return {'origin': origin, 'trans': trans}