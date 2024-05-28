#Imports
from __future__ import print_function, unicode_literals, absolute_import, division
from datetime import datetime
from pathlib import Path
import time
import csv
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import json
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from sklearn.model_selection import train_test_split

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist.utils import mask_to_categorical
from stardist.plot import render_label


#Number of Classes and channels for Identification
n_classes = 2
n_channel=3

#Loading in data where Masks are .tif files and images are .tiff
X = sorted(glob(r'/scratch/hngu247/multiclass_training/Training_Files/content/Embryo/images/*.tiff'))
Y = sorted(glob(r'/scratch/hngu247/multiclass_training/Training_Files/content/Embryo/masks/*.tif'))
assert all(Path(x).name==Path(y).name + "f" for x,y in zip(X,Y))



X = list(map(imread,X))
Y = list(map(imread,Y))

axis_norm = (0,1)
X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
Y = [fill_label_holes(y) for y in Y]

#Convert The data to 3 channels
for i in range(0,len(X)):
    if len(X[i].shape) > 2 and X[i].shape[2] == 4:
    #slice off the alpha channel
        X[i] = X[i][:, :, :3]

# # Import the dictionaries

directory = r'/scratch/hngu247/multiclass_training/Training_Files/content/Embryo/dict'
import os

all_dicts = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Check if the file is a text file
        filepath = os.path.join(directory, filename)

        # Read the dictionary file
        with open(filepath, 'r') as file:
            lines = file.readlines()

        # Create dictionary from lines
        my_dict = {}
        for line in lines:
            line = line.strip()
            if ':' in line:  # Check if the line has the correct format
                parts = line.strip().split(':')
                if len(parts) == 2:
                    label = [s for s in parts[0].split() if s.isdigit()][0]
                    if "0" in parts[1].strip():
                        value = 1 #Non-Viable
                    else:
                        value = 2 #Viable
                my_dict[label] = value
            # else:
                # print(f"Skipping line '{line}' in file '{filename}' as it does not have the correct format.")

        # Add the created dictionary to all_dicts with filename as key
        all_dicts[filename] = my_dict

myKeys = list(all_dicts.keys())
myKeys.sort()
all_dicts = {i: all_dicts[i] for i in myKeys}
 

list_of_dicts = []
updated_data = {}

print(all_dicts)

#Relabel and organize the dictionaries

# Iterate through the dictionary items
for file in all_dicts:
  for key, value in all_dicts[file].items():
    # Remove 'Label' from the keys and convert to integer
    new_key = int(key.replace('Label', '').strip())
    # Add new key with updated value to the new dictionary
    updated_data[new_key] = value
  list_of_dicts.append(updated_data)
  updated_data = {}

#set up the model

# 32 is a good default choice
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
    n_classes    = n_classes,   # set the number of object classes
)
vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(None, allow_growth=True)
    # alternatively, adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    # limit_gpu_memory(0.8)    

model = StarDist2D(conf, name='stardist_multiclass', basedir='models')

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

X_trn, X_val, Y_trn, Y_val, C_trn, C_val = train_test_split(X,Y,list_of_dicts,test_size=.20,random_state = 42)

log_dir = "/scratch/hngu247/multiclass_training/Training_Files/tensorboard_loss"

# for i in steps:
model_name = 'Stardist_' + str(i)
model = StarDist2D(conf, name='stardist_multiclass', basedir='models')
median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))

# log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

model.train(X_trn,Y_trn, classes=C_trn, validation_data=(X_val,Y_val,C_val), augmenter=augmenter, epochs=500) # 200 epochs seem to be enough for synthetic demo dataset

model.optimize_thresholds(X_val, Y_val)

def class_from_res(res):
    cls_dict = dict((i+1,c) for i,c in enumerate(res['class_id']))
    return cls_dict

    # Y_val_pred, res_val_pred = tuple(zip(*[model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)for x in tqdm(X_val[:])]))

    # stats = matching_dataset(Y_val, Y_val_pred, thresh=.5, show_progress=False)
    # print(stats)
    # # making a new csv file
    # filename = os.path.join(model.basedir, model_name, 'val_stats.csv')
    
    # # Save the data to the CSV file
    # with open(filename, 'w', newline='') as csvfile:
    #     fieldnames = ['criterion', 'thresh', 'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality', 'by_image']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     # Write the header row
    #     writer.writeheader()

    #     # Write each DatasetMatching object as a row in the CSV file
    #     writer.writerow({
    #             'criterion': stats.criterion,
    #             'thresh': stats.thresh,
    #             'fp': stats.fp,
    #             'tp': stats.tp,
    #             'fn': stats.fn,
    #             'precision': stats.precision,
    #             'recall': stats.recall,
    #             'accuracy': stats.accuracy,
    #             'f1': stats.f1,
    #             'n_true': stats.n_true,
    #             'n_pred': stats.n_pred,
    #             'mean_true_score': stats.mean_true_score,
    #             'mean_matched_score': stats.mean_matched_score,
    #             'panoptic_quality': stats.panoptic_quality,
    #             'by_image': stats.by_image,
    #         })




# model.train(X_trn,Y_trn, classes=C_trn, validation_data=(X_val,Y_val,C_val), augmenter=augmenter, epochs=200) # 200 epochs seem to be enough for synthetic demo dataset

# model.optimize_thresholds(X_val, Y_val)

# i = 0
# label, res = model.predict_instances(X_val[i], n_tiles=model._guess_n_tiles(X_val[i]))

# def class_from_res(res):
#     cls_dict = dict((i+1,c) for i,c in enumerate(res['class_id']))
#     return cls_dict

# print(class_from_res(res))

# Y_val_pred, res_val_pred = tuple(zip(*[model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)
#               for x in tqdm(X_val[:])]))

# stats = matching_dataset(Y_val, Y_val_pred, thresh=.5, show_progress=False)
# print(stats)