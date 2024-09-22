# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   train.py
@Time    :   2023/08/10 02:02:35
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''

#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import numpy as np
from model import *
from preprocess import *
import argparse
from tqdm import tqdm
import logging
import os


# %% datapath
train_path = "data/train"
valid_path = "data/valid"
image_folder = "images"
label_folder = "label"

# %% create train and validation data list
train_img_path = os.path.join(train_path,image_folder)
train_label_path = os.path.join(train_path,label_folder)
valid_img_path = os.path.join(valid_path,image_folder)
valid_label_path = os.path.join(valid_path,label_folder)

train_list = sorted(os.listdir(train_img_path))
valid_list = sorted(os.listdir(valid_img_path))
#%% parameters
parser = argparse.ArgumentParser(description='PyTorch Unet')
parser.add_argument('--batch_size', type=int, default = 4, metavar='N',
                        help='input batch size for training (default: 4)')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 50)')
parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--save_folder', default='checkpoints/',
                        help='Directory for saving checkpoint models')
parser.add_argument('--load_model', default=False,
                        help='load saved models')
args = parser.parse_args(args=[])
if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
#%% get data means and variance through the preprocessing module
mean = [0.5703, 0.5849, 0.5087, 0.6726]
std = [0.1768, 0.1780, 0.2328, 0.1719]

# %% create train datasets 
train_dataset = UnetDataset(img_root = train_img_path,
                            label_root = train_label_path,
                            img_list=train_list,
                            label_list=train_list,
                            transform=None,  # You can add any additional transformations if needed
                            target_transform=None,
                            mean=mean,
                            std=std,
                            train=True)

# %% create valid datasets
val_dataset = UnetDataset(img_root = valid_img_path,
                          label_root = valid_label_path,
                          img_list=valid_list,
                          label_list=valid_list,
                          transform=None,  # You can add any additional transformations if needed
                          target_transform=None,
                          mean=mean,
                          std=std,
                          train=False)

# import matplotlib.pyplot as plt
# image,label = val_dataset[29]
# plt.subplot(1,2,1)
# plt.imshow(image.squeeze()[0])
# plt.subplot(1,2,2)
# plt.imshow(label.squeeze())
# plt.show()

# %% create dataloader
kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}
train_loader = DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = DataLoader(val_dataset,
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

# for image, label in train_loader:
#     print(image.shape)
#     print(label.shape)

# %%
model = UNetplusplus(4,4,deep_supervision=False).cuda()
if args.load_model:
    state_dict = torch.load('checkpoints/last_model.pth', map_location=device)
    model.load_state_dict(state_dict)
model = model.to(device)
optimizer = torch.optim.NAdam(model.parameters(), # 比Adam效果好一点，收敛快，但loss乱跳
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0)

# optimizer = torch.optim.SGD(model.parameters(), # loss下降均匀，收敛慢，有潜力，可以一试
#                 lr=args.lr,
#                 momentum=0.8,
#                 weight_decay=0)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       'min',factor=0.35,
                                                       threshold=0.0001, 
                                                       verbose=1,min_lr=0.0001,
                                                       patience=10)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Loss Function
criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Controls the weight of balancing positive and negative samples
        self.gamma = gamma  # Controls how much to focus on hard samples
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)  
        p_t = torch.exp(-ce_loss)  # compute probability
        focal_loss = self.alpha * (1 - p_t)**self.gamma * ce_loss  # compute Focal Loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
focal_loss_fn = FocalLoss(alpha=0.5, gamma=2, reduction='mean')
# DiceLoss
def dice_coeff(input, target, epsilon=1e-6):
    assert input.size() == target.size()

    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target)

    if sets_sum.item() == 0:
        sets_sum = 2 * inter
    return (2. * inter + epsilon) / (sets_sum + epsilon)


def multiclass_dice_coeff(input, target, epsilon=1e-6):
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]): 
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], epsilon)
    return dice / (input.shape[1])  

def dice_loss(input, target, multiclass=False):
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target) 


# function to calculate accuracy 
def calculate_accuracy(pred_masks, true_masks):
    pred_masks = torch.softmax(pred_masks, dim=1)
    pred_masks = torch.argmax(pred_masks, dim=1)
    accuracy = torch.mean((pred_masks == true_masks).float())

    return accuracy.item()

# %%
# Cutmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(beta, input, target):
    # generate mixed sample
    lam = np.random.beta(beta, beta)  

    rand_index = torch.randperm(input.size()[0]).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)  

    input_clone = input.clone()
    target_clone = target.clone()
    input_clone[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

    target_clone[:, :, bbx1:bbx2, bby1:bby2] = target[rand_index, :, bbx1:bbx2, bby1:bby2]
        
    return input_clone, target_clone

# %%
# Define a logging to log training progress
logging.basicConfig(filename='temp.log', level=logging.INFO, format='%(message)s', filemode='w')
# Add header to the log file
header = "Epoch,Loss,Train Focal Loss,Train Dice Loss,Train Accuracy,Validation Focal Loss,Validation Dice Loss,Validation Loss,Validation Accuracy"
logging.info(header)
# Training loop
best_accuracy = 0.0
for epoch in range(1, args.epochs + 1):
    model.to(device)
    model.train()
    train_loss = 0.0
    train_fl_loss = 0.0
    train_dice_loss = 0.0
    train_accuracy = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100) as pbar:
        for batch_idx, (image, true_masks) in enumerate(pbar):
            image = image.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.long)
           
            # cutmix
            # input, target = cutmix(1.0, image, true_masks)
            input, target = image, true_masks
            # Forward pass
            output = model(input) # batchsize * numclass * 128 * 128
            target = target.squeeze(1) # batchsize * 128 * 128
                
            # Calculate the loss
            fl_loss_train = focal_loss_fn(output, target)
            dice_loss_train = dice_loss(
                F.softmax(output, dim=1).float(), # batchsize * numclass * 128 * 128
                F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float(), # batchsize * numclass * 128 * 128
                multiclass=True
            )

            # Backward pass and optimization
            loss = fl_loss_train + dice_loss_train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_fl_loss += fl_loss_train.item()
            train_dice_loss += dice_loss_train.item()
            train_accuracy += calculate_accuracy(output, target)
            
            # Update tqdm progress bar
            pbar.set_postfix({"Loss": train_loss / (batch_idx + 1), "Dice Loss": train_dice_loss / (batch_idx + 1),"Train Accuracy": train_accuracy / (batch_idx + 1)})
        
        # Logging and saving the model parameters after each epoch
        train_loss /= len(train_loader)
        train_fl_loss /= len(train_loader)
        train_dice_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        
# Validation loop
    model.eval()
    val_loss = 0.0
    val_fl_loss = 0.0
    val_dice_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad(), tqdm(val_loader, desc="Validation") as pbar:
        for batch_idx, (image, true_masks) in enumerate(pbar):
            image = image.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.long)
            
            # Forward pass
            output = model(image) # batchsize * numclass * 128 * 128
            true_masks = true_masks.squeeze(1) # batchsize * 128 * 128

            # Calculate the loss
            fl_loss_val = focal_loss_fn(output, true_masks)
            dice_loss_val = dice_loss(
                F.softmax(output, dim=1).float(), # batchsize * numclass * 128 * 128
                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(), # batchsize * numclass * 128 * 128
                multiclass=True
            )
            loss = fl_loss_val + dice_loss_val

            val_loss += loss.item()
            val_fl_loss += fl_loss_val.item()
            val_dice_loss += dice_loss_val.item()
            val_accuracy += calculate_accuracy(output, true_masks)

            pbar.set_postfix({"Val Loss": val_loss / (batch_idx + 1),"Val Dice Loss": val_dice_loss / (batch_idx + 1), "Val Accuracy": val_accuracy / (batch_idx + 1)})
        
        val_loss /= len(val_loader)
        val_fl_loss /= len(val_loader)
        val_dice_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        
        
        # Update the learning rate using the scheduler
        scheduler.step(metrics=val_accuracy)
        
    log_line = f"{epoch}, {train_loss:.10f}, {train_fl_loss:.10f}, {train_dice_loss:.10f}, {train_accuracy:.10f}, {val_fl_loss:.10f}, {val_dice_loss:.10f}, {val_loss:.10f},  {val_accuracy:.10f}"
    logging.info(log_line)

    # Save the model with the best validation accuracy
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), args.save_folder + f"best_model.pth")
        print("======Saving model successfully======")
    if epoch == args.epochs:
        torch.save(model.state_dict(), args.save_folder + f"last_model.pth")
        print("======Saving model successfully======")
print("Training completed.")

# %% save log file
if args.load_model:
    # Append subsequent training logs to the same file
    log_folder = 'log'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    log_file = os.path.join(log_folder, 'training.log')
    with open(log_file, 'a') as f0:
        with open('temp.log', 'r') as f:
            log_content = f.readlines()[1:]  # Skip the CSV file's column names
            for line in log_content:
                f0.write(line.strip() + '\n')
else:
    # Load the first training log file
    log_folder = 'log'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    log_file = os.path.join(log_folder, 'training.log')
    with open(log_file, 'w') as f:
        with open('temp.log', 'r') as f0:
            log_content = f0.readlines()
            for line in log_content:
                f.write(line)