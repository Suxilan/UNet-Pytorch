from torch.utils.data import Dataset
from .transform import *
import os

# %% difine the unet dataset
class UnetDataset(Dataset):
    def __init__(self, img_root, label_root,
                 img_list=None, label_list=None,
                 transform=None, target_transform=None,
                 mean=None, std=None, train=True):
        assert img_root is not None and label_root is not None, 'Must specify img_root and label_root!'
        self.img_root = img_root
        self.label_root = label_root
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.target_transform = target_transform
        self.mean = mean
        self.std = std
        self.train = train

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))

        if self.transform is not None:
                image = self.transform(image)
                
        if self.label_list is not None:
            label_name = self.label_list[index].replace(".tif", ".png")
            label = Image.open(os.path.join(self.label_root, label_name))

            if self.train:
                transforms = SegmentationTransforms(self.mean, self.std)
                image, label = transforms((image, label))
            else:
                transforms = SegmentationTransforms(self.mean, self.std)
                image, label = transforms.valid_transform((image, label))

            if self.target_transform is not None:
                label = self.target_transform(label)
                
            return image, label
        else:
            if self.transform is not None:
                image = self.transform(image)
            return image

    def __len__(self):
        return len(self.img_list)
