# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   classify.py
@Time    :   2023/08/11 16:15:40
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''

import sys
import os
o_path = os.getcwd() 
sys.path.append(o_path) 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

#%% Load the image
img_path = "03_predict.tif"
image = Image.open(img_path)

# Define class labels
classes = ["Non-Vegetation", "Wheat", "Rape", "Other Vegetation"]

num_rows, num_cols = image.size

# Generate random latitude and longitude values for the example
lats = np.linspace(40.0, 41.0, num_rows)
lons = np.linspace(-105.0, -104.0, num_cols)

# Set seaborn style and ggplot context
sns.set(style="white")
sns.set_context("notebook", rc={"axes.labelsize": 14, "axes.titlesize": 16})

# Create a colormap for the classes
cmap = plt.get_cmap("tab20", len(classes))

# Plot the classification map
plt.figure(figsize=(12, 8))
plt.imshow(image, cmap=cmap, extent=[lons.min(), lons.max(), lats.min(), lats.max()])

# Create a scatter plot for the legend
for i, label in enumerate(classes):
    plt.scatter([], [], c=[cmap(i)], label=label, s=100)

# Add a legend
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper left')

# Add a colorbar
cbar = plt.colorbar()
cbar.set_ticks(np.arange(len(classes)))
cbar.set_ticklabels(classes)
cbar.set_label("Class")

# Set x and y labels
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Add a north arrow
plt.annotate("N", xy=(0.5, 0.95), xycoords="axes fraction",
             fontsize=16, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Set title
plt.title("Land Cover Classification Map")

# Hide axes
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
# %%
