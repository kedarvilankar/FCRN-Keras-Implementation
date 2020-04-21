# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:22:23 2019

@author: kedar


Generate depth maps for SYNS test dataset and store it in outputDir
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import scipy.io as sio
#import ResNet50 as rn50
from FCRN import reverseHuber, coeff_determination
from tensorflow.keras.models import Model, load_model

#from tqdm import tqdm
# test set path
DATADIR = "D:/SYNS_Dataset_ForDNN3/test_Data/"
outputDir = 'D:/SYNS_Dataset_ForDNN3/test_Data/output/FCRN_ResnetPreTrained_SYNS_adam3/'

dataDir_pickle = 'D:/SYNS_Dataset_ForDNN3/'
# load pickle files
pickle_in = open(dataDir_pickle + "test_X.pickle","rb")
X_test = pickle.load(pickle_in)
    
pickle_in = open(dataDir_pickle + "test_Y.pickle","rb")
Y_test = pickle.load(pickle_in)
Y_test = np.expand_dims(Y_test, axis=3) 


filesInDIR = os.listdir(DATADIR)

numFiles = len(filesInDIR)
#for img in os.listdir(DATADIR)
numImageFiles = int(numFiles/2)


#model = rn50.ResNet50_Pretrained(input_shape = (300, 383, 3))
#model.load_weights('output/FCRN-SYNS_FlipAug_Resnet_Pretrained-best.hdf5')

# load the model
custom_objects={'reverseHuber': reverseHuber, 'coeff_determination': coeff_determination}
model = load_model('output/FCRN-SYNS_FlipAug_Resnet_Pretrained_optimizer_adam3-best.hdf5', custom_objects = custom_objects)

preds = model.evaluate(X_test, Y_test)
preds = model.predict(X_test)


# generate depth maps for image crops in test set
for fileNo in np.arange(1,numImageFiles+1):
    imgFileName = DATADIR + str(fileNo).zfill(8) + '_colors.png'
    depthFileName = DATADIR + str(fileNo).zfill(8) + '_depth.png'
    
    img_array = cv2.imread(imgFileName)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    
#    plt.imshow(img_array)
#    plt.show
    
    img_array =  np.expand_dims(img_array, axis=0) 
    y_pred = model.predict(img_array)
    
#    plt.figure()
#    plt.imshow(np.squeeze(y_pred))
#    plt.show
    
    
    result = {}
    result['depth'] = np.squeeze(y_pred)
    sio.savemat(outputDir + str(fileNo).zfill(8) + '_pred.mat', result)
    
#    break
    
