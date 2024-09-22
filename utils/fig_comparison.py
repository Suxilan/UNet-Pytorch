# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   comparision.py
@Time    :   2023/08/11 13:32:33
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''
import sys
import os
o_path = os.getcwd() 
sys.path.append(o_path) 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import *

#%% 
img_path = "data/valid/images/008.tif"
label_path = "data/valid/label/008.png"

image = Image.open(img_path)
label = Image.open(label_path)

mean = [0.5703, 0.5849, 0.5087, 0.6726]
std = [0.1768, 0.1780, 0.2328, 0.1719]

#%%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
tensor_image = transform(image)
tensor_image = tensor_image.unsqueeze(0)
use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#%%
def load_model_and_test(modelname,net):
    global tensor_image
    model = net
    state_dict = torch.load('checkpoints/'+modelname+'.pth', map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    tensor_image = tensor_image.to(device, dtype=torch.float32)
    output = model(tensor_image)
    output = torch.softmax(output, dim=1)

    return output
# %%
def display(modelList,outputList):
    global tensor_image, label
    def colormap():
        cdict = np.array([(255, 239, 170), (1, 148, 16), (192, 255, 141), (1, 255, 141)]) / 255
        return colors.ListedColormap(cdict, 'indexed', 4)

    my_cmap = colormap()
    classes = ["Non-Vegetation", "Wheat", "Rape", "Other Vegetation"]

    num_models = len(modelList)
    sns.set_style("whitegrid")  

    fig, axes = plt.subplots(num_models, 3, figsize=(22, 6 * num_models))
    for i in range(num_models):
        output = torch.argmax(outputList[i], dim=1)
        output = output.squeeze().cpu().numpy()

        axes[i, 0].imshow(image, cmap=my_cmap,interpolation="bilinear")
        axes[i, 0].set_title('Image' if i == 0 else '',fontsize = 16, fontweight='bold',pad = 15)
        axes[i, 0].axis('off')  # Turn off axis

        axes[i, 1].imshow(label, cmap=my_cmap, interpolation="bilinear")
        axes[i, 1].set_title('Label' if i == 0 else '',fontsize = 16, fontweight='bold', pad = 15)
        axes[i, 1].axis('off')  # Turn off axis

        axes[i, 2].imshow(output, cmap=my_cmap, interpolation="bilinear")
        axes[i, 2].set_title(modelList[i],fontsize = 16, fontweight='bold',pad = 15)
        axes[i, 2].axis('off')  # Turn off axis

        for j in range(len(classes)):
            axes[i, 2].scatter([], [], c=np.array(my_cmap(j)).reshape(1, -1), s=100, label=classes[j])  

    # Set a single legend for all subplots at the bottom
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.08), ncol=len(classes), fontsize = 30,) 

    #plt.tight_layout()
    # plt.subplots_adjust(wspace=0.5) 

    plt.savefig("fig/model_comparison.png", dpi=300,bbox_inches="tight")
    plt.show()
    
#%%
if __name__=='__main__':
    modelList = ['UNet',
                'UNet++',
                'ResNet101+DeepLabV3+',
                'UNet++ & CutMix',
                'UNet & CutMix']
    modelList = ['best_model',
                'best_model',
                'best_model',
                'best_model',
                'best_model']
    nets = [UNet(4,4).cuda(),
            UNet(4,4).cuda(),
            UNet(4,4).cuda(),
            UNet(4,4).cuda(),
            UNet(4,4).cuda()]
    outputList = []
    for i in range(len(modelList)):
        output = load_model_and_test(modelList[i],nets[i])
        outputList.append(output)

    display(modelList,outputList)
# %%
