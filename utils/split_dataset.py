# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   split_dataset.py
@Time    :   2023/08/07 18:07:40
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''

import sys
import os
o_path = os.getcwd() 
sys.path.append(o_path) 
import random
import shutil

# Define the data path
data_path = "data"
train_data_path = "data/train"
valid_data_path = "data/valid"

# Create the 'train' folder if it doesn't exist
os.makedirs(train_data_path, exist_ok=True)
# Create the 'valid' folder if it doesn't exist
os.makedirs(valid_data_path, exist_ok=True)

# Create 'images' and 'label' folders inside 'valid' folder
train_img_folder = os.path.join(train_data_path, "images")
train_label_folder = os.path.join(train_data_path, "label")
os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)

# Create 'images' and 'label' folders inside 'valid' folder
valid_img_folder = os.path.join(valid_data_path, "images")
valid_label_folder = os.path.join(valid_data_path, "label")
os.makedirs(valid_img_folder, exist_ok=True)
os.makedirs(valid_label_folder, exist_ok=True)

# Get the list of images and labels from the 'train' folder
ori_img_path = os.path.join(data_path, "images")
ori_list = sorted(os.listdir(ori_img_path))
split_index = int(0.8 * len(ori_list))
random.seed(42)
random.shuffle(ori_list)
train_img_list, val_img_list = ori_list[:split_index], ori_list[split_index:]

ori_label_path = os.path.join(data_path, "label")
label_list = sorted(os.listdir(ori_label_path))
random.seed(42)
random.shuffle(label_list)
train_label_list, val_label_list = label_list[:split_index], label_list[split_index:]

# Move the selected validation data to the 'valid' folder and renumber the files
def move_and_renumber1(src_folder, dest_folder, file_list):
    for idx, filename in enumerate(file_list):
        src = os.path.join(src_folder, filename)
        dest = os.path.join(dest_folder, f"{idx:03d}.tif")  # Renaming with three-digit padding
        shutil.move(src, dest)

def move_and_renumber2(src_folder, dest_folder, file_list):
    for idx, filename in enumerate(file_list):
        src = os.path.join(src_folder, filename)
        dest = os.path.join(dest_folder, f"{idx:03d}.png")  # Renaming with three-digit padding
        shutil.move(src, dest)
        
# renumber the remaining training data
move_and_renumber1(ori_img_path, train_img_folder, train_img_list)
move_and_renumber2(ori_label_path, train_label_folder, train_label_list)  
      
# Move and renumber the validation data
move_and_renumber1(ori_img_path, valid_img_folder, val_img_list)
move_and_renumber2(ori_label_path, valid_label_folder, val_label_list)

        
print("Validation data moved to the 'valid' folder and dataset renumbered.")
