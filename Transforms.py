import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class ResizeTo(torch.nn.Module):
    """
    This transform resizes max(width, height) to 256px
    :param dim: The dimension to which the image will be resized
    """

    def __init__(self, dim=256, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        :param img: an RGB PIL image
        :return:
        """

        width, height = img.size

        if width > height:
            new_width = self.dim
            new_height = int(self.dim * (height / width))
            return F.resize(img, [new_height, new_width], self.interpolation, self.max_size, self.antialias)
        elif width < height:
            new_width = int(self.dim * (width / height))
            new_height = self.dim
            return F.resize(img, [new_height, new_width], self.interpolation, self.max_size, self.antialias)
        else:
            return F.resize(img, [self.dim, self.dim], self.interpolation, self.max_size, self.antialias)


class PadTo(torch.nn.Module):
    """
    This transform resizes max(width, height) to 256px
    :param dim: The dimension to which the image will be padded
    """

    def __init__(self, dim=256, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):

        width, height = img.size

        # Handling odd dimensions
        if width % 2 != 0:
            # img = transforms.Pad((1, 0))(img)
            img = F.pad(img, [1, 0, 0, 0])
            width += 1
        if height % 2 != 0:
            # img = transforms.Pad((0, 1))(img)
            img = F.pad(img, [0, 1, 0, 0])
            height += 1

        # Transforming
        if width > height:
            x = 0
            y = int((self.dim - height)/2)
            return transforms.Pad((x, y))(img)
        elif width < height:
            x = int((self.dim - width)/2)
            y = 0
            return transforms.Pad((x, y))(img)
        else:
            x = int((self.dim - width)/2)
            y = x
            return transforms.Pad((x, y))(img)


def create_transform(size=256):
    """
    Create a composition of transforms that:
        1. Resizes the image without distortion max(width, height)=size
        2. Pads the image to dimensions (size, size)
        3. Normalizes pixel values of each channel (mean= , std=)

    :param size: single integer value
    :return: composition of transforms
    """
    return transforms.Compose([
        ResizeTo(size),
        PadTo(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def create_transform_augment(size=256):
    """
    Create a composition of transforms that:
        1. Augments the image using random rotation and color jitter
        2. Resizes the image without distortion max(width, height)=size
        3. Pads the image to dimensions (size, size)
        4. Normalizes pixel values of each channel (mean= , std=)

    :param size: single integer value
    :return: composition of transforms
    """
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=30),
        ResizeTo(size),
        PadTo(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
