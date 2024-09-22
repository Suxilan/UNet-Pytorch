import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from model import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

ori_img_path = "data/test/images"
test_img_list = sorted(os.listdir(ori_img_path))
ori_label_path = "data/test/label"
test_label_list = sorted(os.listdir(ori_label_path))


class UnetTestDataset(Dataset):
    """
    You need to inherit nn.Module and
    overwrite __getitem__ and __len__ methods.
    """
    def __init__(self, img_root=ori_img_path,
                 img_list=None, transform=None):
        assert img_root is not None, 'Must specify img_root and label_root!'

        self.img_root = img_root
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        image_name = self.img_list[index][:-4]
        if self.transform is not None:
            image = self.transform(image)

        return image, image_name

    def __len__(self):
        return len(self.img_list)
    
if not os.path.isdir('results/'):
    os.mkdir('results/')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
mean = [0.5703, 0.5849, 0.5087, 0.6726]
std = [0.1768, 0.1780, 0.2328, 0.1719]
    
transform_image = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_dataset = UnetTestDataset(img_list=test_img_list, transform=transform_image)
test_loader = DataLoader(test_dataset,
    batch_size=1, shuffle=False, num_workers=4)

model = UNet(4,4).cuda()
model.load_state_dict(torch.load('./checkpoints/best_model.pth'))
model = model.to(device)
pic = []

def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, name) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            output = torch.softmax(output, dim=1)
            
            # print(output.shape)
            pic.append((output, name))
    return pic

def save_output_image(output, filename):
    plt.imsave(filename, output)

def display(results):
    cdict = np.array([(255,239,170),(1,148,16),(192,255,141),(1,255,141)])/255  #,(250,141,52),(40,146,199)
    def colormap():
        return colors.ListedColormap(cdict, 'indexed',4)
    my_cmap = colormap()
    classes = ["其他","小麦","油菜","其他植被"]
    for i in range(len(results)):
        label = Image.open(os.path.join(ori_label_path, test_label_list[i]))
        plt.subplot(3,4,2*i+1)
        plt.imshow(results[i], cmap=my_cmap)
        plt.title(test_img_list[i]+'地物分类结果图')
        
        plt.subplot(3,4,2*i+2)
        plt.imshow(label, cmap=my_cmap)
        plt.title(test_img_list[i]+' label')
        clb=plt.colorbar()

        for j in range(len(cdict)):
            plt.scatter([],[],c=np.array(cdict[j]).reshape(1,-1),s=50,label=classes[j])
        
        plt.legend(loc='lower right')
    plt.show()

import cv2
def main():
    test(model, device, test_loader)
    i=0
    results = []
    for img_tensor in pic:
        img = torch.argmax(img_tensor[0], dim=1)
        img = img.squeeze().cpu().numpy()
        results.append(img)
        i = i+1
    display(results)

if __name__ == '__main__':
    main()