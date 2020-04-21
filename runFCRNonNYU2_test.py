# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:01:12 2019

@author: kedar

Generate depth maps for NYU2 test dataset and store it in outputDir
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import glob
import ntpath
import scipy.io as sio
#import ResNet50 as rn50
from FCRN import reverseHuber, coeff_determination
from tensorflow.keras.models import Model, load_model
from skimage.transform import resize
from skimage.io import imread


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
#from tqdm import tqdm
# test set path
DATADIR = "D:/NYU_dataSet/data/nyu2_test/"
outputDir = "D:/NYU_dataSet/data/nyu2_test/predMaps/"
chkPoint = 'C:/Users/kedar/OneDrive - York University/Documents/singleView3D_DNN/FCRN-Keras/output/FCRN-NYU2_reTrained_adam-best.hdf5'

filesInDIR = os.listdir(DATADIR)

numFiles = len(filesInDIR)
#for img in os.listdir(DATADIR)
numImageFiles = int(numFiles/2)
colorImgFiles = []
for file in glob.glob(DATADIR+"*_colors.png"):
    colorImgFiles.append(path_leaf(file))


# load the model
model = load_model(chkPoint)


# generate depth maps for image crops in test set
for afile in colorImgFiles:
    imgFileName = DATADIR + afile
    depthFileName = DATADIR + afile[0:5] + "_depth.png"
    
    img_array = resize(imread(imgFileName), (300, 383))
    
    y_gt = resize(imread(depthFileName), (194,258))
    y_gt = np.expand_dims(y_gt, axis=0)
    y_gt = np.expand_dims(y_gt, axis=3)
    
    #cv2.imshow('color',img_array,  )
#    plt.imshow(img_array)
#    plt.show
    
    img_array =  np.expand_dims(img_array, axis=0) 
    preds = model.evaluate(img_array, y_gt)
    y_pred = model.predict(img_array)
    
    #result = {}
    #result['depth'] = np.squeeze(y_pred)
    #sio.savemat(outputDir + afile[0:5] + '_pred.mat', result)
   
