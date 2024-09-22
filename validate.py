# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   validate.py
@Time    :   2023/08/07 20:17:44
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''

from model import *
from preprocess import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score

#%%
valid_path = "data"
image_folder = "images"
label_folder = "label"
valid_img_path = os.path.join(valid_path,image_folder)
valid_label_path = os.path.join(valid_path,label_folder)
valid_list = sorted(os.listdir(valid_img_path))
mean = [0.5703, 0.5849, 0.5087, 0.6726]
std = [0.1768, 0.1780, 0.2328, 0.1719]
val_dataset = UnetDataset(img_root = valid_img_path,
                          label_root = valid_label_path,
                          img_list=valid_list,
                          label_list=valid_list,
                          transform=None,  # You can add any additional transformations if needed
                          target_transform=None,
                          mean=mean,
                          std=std,
                          train=False)
val_loader = DataLoader(val_dataset,
    batch_size=1, shuffle=False, num_workers=0,pin_memory=False)
#%%
use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = UNetplusplus(4,4,deep_supervision=False).cuda()
state_dict = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)

#%%
# Set the model to evaluation mode
model.eval()

# Initialize variables to store metrics
y_true = []
y_pred = []

with torch.no_grad(), tqdm(val_loader, desc="Validation") as pbar:
    for batch_idx, (image, true_masks) in enumerate(pbar):
        image = image.to(device, dtype=torch.float32)
        true_masks = true_masks.to(device, dtype=torch.long)
        
        # Forward pass
        output = model(image)
        true_masks = true_masks.squeeze(1)

        # Calculate validation metrics for each batch
        output = torch.softmax(output, dim=1)
        predicted_masks = torch.argmax(output, dim=1)

        y_true.extend(true_masks.view(-1).cpu().numpy())
        y_pred.extend(predicted_masks.view(-1).cpu().numpy())
#%%
def dice_coefficient(y_true, y_pred, class_id):
    intersection = np.sum((y_true == class_id) & (y_pred == class_id))
    total_pixels_gt = np.sum(y_true == class_id)
    total_pixels_pred = np.sum(y_pred == class_id)
    
    dice = (2 * intersection) / (total_pixels_gt + total_pixels_pred)
    return dice

def calculate_iou(y_true, y_pred, num_classes):
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    intersection = np.diag(confusion)
    union = np.sum(confusion, axis=0) + np.sum(confusion, axis=1) - intersection
    iou = intersection / union
    return iou

def calculate_miou(y_true, y_pred, num_classes):
    iou = calculate_iou(y_true, y_pred, num_classes)
    miou = np.mean(iou)
    return miou
#%%
#Calculate validation metrics
val_accuracy = accuracy_score(y_true, y_pred)

y_true = [y_true[i] if y_true[i] == 1 else 0 for i in range(len(y_true))]
y_pred = [y_pred[i] if y_pred[i] == 1 else 0 for i in range(len(y_pred))]
#%%
class_ids = [0, 1]
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=class_ids)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=class_ids)

# Calculate kappa coefficient
kappa = cohen_kappa_score(y_true, y_pred)

# Calculate dice coefficient
dice = dice_coefficient(np.array(y_true), np.array(y_pred), 1)
miou = calculate_miou(y_true, y_pred, 2)
print("mIoU:", miou)
print("Validation Accuracy: {:.4f}".format(val_accuracy))
print("dice: {:.4f}".format(dice))
# print("Average Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1_score[1]))
print("Confusion Matrix:")
print(conf_matrix)
print("Kappa Coefficient: {:.4f}".format(kappa))

#--------------------------------可视化-------------------------------------
# 打印输出结果
# %% painting
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['Arial'] 

if not os.path.isdir('fig/'):
    os.mkdir('fig/')
# Set the global style for all the plots
sns.set_style("whitegrid")
plt.style.use('seaborn-whitegrid')

# Load the training log data
training_log = pd.read_csv('log/training.log')

# Define the color palette for the plots
palette = sns.color_palette('pastel', n_colors=3)
color_map = colors.LinearSegmentedColormap.from_list("", palette)

# Plot the training and validation loss curves
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1, len(training_log["Epoch"]) + 1), training_log["Loss"], label="Training Loss")
plt.plot(range(1, len(training_log["Epoch"]) + 1), training_log["Validation Loss"], label="Validation Loss")
plt.title("Training and Validation Loss", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="upper right", fontsize=12)
plt.savefig("fig/loss.png", dpi=300, bbox_inches='tight')

# %%
# Plot the accuracy curves
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1, len(training_log["Epoch"]) + 1), training_log["Train Accuracy"], label="Train Accuracy")
plt.plot(range(1, len(training_log["Epoch"]) + 1), training_log["Validation Accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.savefig("fig/acc.png", dpi=300, bbox_inches='tight')

# %%
# Plot the dice score curves
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1, len(training_log["Epoch"]) + 1), 1-training_log["Train Dice Loss"], label="Train Dice Score")
plt.plot(range(1, len(training_log["Epoch"]) + 1), 1-training_log["Validation Dice Loss"], label="Validation Dice Score")
plt.title("Training and Validation Dice Scores", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Dice Score", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.savefig("fig/dice.png", dpi=300, bbox_inches='tight')

# %%
# Plot the confusion matrix heatmap
labels = ['other', 'Wheat', 'Rapeseed', 'other weed']

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, cmap='Greens', annot=True, fmt='d', square=True, cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("True Class", fontsize=14)
title = ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
title.set_position([.6, 1.15])
plt.savefig("fig/Confusion Matrix.png", dpi=300, bbox_inches='tight')

#%%
# Plot the results
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width, f1_score, width, label='F1 Score', color=color_map(0.2))
rects2 = ax.bar(x, recall, width, label='Recall', color=color_map(0.5))
rects3 = ax.bar(x + width, precision, width, label='Precision', color=color_map(0.8))

# Add kappa label to the title
title = f'Evaluations for Each Class (Kappa: {kappa:.2f})'
ax.set_title(title, fontsize=16, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Metric', fontsize=14)
ax.set_ylim(0, 1.3)
ax.grid(axis='y', linestyle='--')
ax.legend(loc='upper right', fontsize=12)

# Add the values to the bars
for rect in rects1+rects2+rects3:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height+0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

# Adjust tick parameters
ax.tick_params(axis='both', which='both', length=0, labelsize=12)

plt.savefig("fig/Evaluations for Each Class.png", dpi=300, bbox_inches='tight')


# %%
