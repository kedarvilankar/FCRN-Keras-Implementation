import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, \
    ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, \
        MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
#from keras.utils import layer_utils
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.densenet import preprocess_input
import pydot
from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
from resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import pickle
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from time import time 
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K
from tensorflow.keras import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50

#identity_block

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), 
               padding = 'valid', name = conv_name_base + '2a', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), 
               padding = 'same', name = conv_name_base + '2b', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), 
               padding = 'valid', name = conv_name_base + '2c', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


#  convolutional_block

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding='same', 
               name = conv_name_base + '2b', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), padding='valid',
               name = conv_name_base + '2c', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), 
                        name = conv_name_base + '1', 
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


# up-convolution block
def upConv_block(X, conv_channels, stage, block):    
    X = UpSampling2D()(X)
    X = Conv2D(conv_channels, (5, 5), name = 'upConv_stage' + str(stage) + '_block' + str(block),
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    return X


def coeff_determination(y_true, y_pred):
#    tmp = K.get_value(y_true)
#    y_true = y_true.flatten()
#    y_pred = y_pred.flatten()
#    idx = np.where(y_true>0)
#    y_true = y_true[idx]
#    y_pred = y_pred[idx]
#    y_true = K.gather(y_true, tf.where(y_true>0))
#    y_pred = K.gather(y_pred, tf.where(y_true>0))
    
    #cond  = y_true>0
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def reverseHuber(y_true, y_pred):
    error = y_true - y_pred
    absError = tf.keras.backend.abs(error)
    
    maxError = tf.keras.backend.max(absError)
    clip_delta = maxError*0.2
    
    cond  = absError <= clip_delta
    squared_Error = (tf.keras.backend.square(error) + tf.keras.backend.square(clip_delta))/ (2*clip_delta)
    return tf.where(cond, absError, squared_Error)
    
    
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))    

def shuffleData(X,Y):
    noOfExamples = X.shape[0]
    shuffledIdx = np.arange(noOfExamples)
    np.random.shuffle(shuffledIdx)
    X_tmp = X[shuffledIdx,:,:,:]
    Y_tmp = Y[shuffledIdx,:,:,:]
    return X_tmp,Y_tmp

#  ResNet50

def ResNet50_manual(input_shape = (228, 304, 3), dropoutRate = 0.3):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    #X = Dropout(rate=dropoutRate)(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = Dropout(rate=dropoutRate)(X)

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = Dropout(rate=dropoutRate)(X)

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = Dropout(rate=dropoutRate)(X)

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = Dropout(rate=dropoutRate)(X)

# =============================================================================
#     # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
#     X = AveragePooling2D(pool_size=(2, 2))(X)
#     
# 
#     # output layer
#     X = Flatten()(X)
#     X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
# =============================================================================
    
    #stage 6 
    X = Conv2D(1024, (1, 1), name = 'conv6', kernel_initializer = glorot_uniform(seed=0))(X)
    X = upConv_block(X, 512, stage = 6, block=1)
    X = upConv_block(X, 256, stage = 6, block=2)
    X = upConv_block(X, 128, stage = 6, block=3)
    X = upConv_block(X, 64, stage = 6, block=4)
    X = upConv_block(X, 32, stage = 6, block=5)
    
    #stage 7
    X = Conv2D(1, (3, 3), name = 'conv7', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


def ResNet50_Pretrained(input_shape = (300, 383, 3)):
    # create the base pre-trained model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None)
    
    X = base_model.output
    #X = AveragePooling2D(pool_size=(2, 2))(X)
    
    #stage 6 
    X = Conv2D(1024, (1, 1), name = 'conv6', kernel_initializer = glorot_uniform())(X)
    X = upConv_block(X, 512, stage = 6, block=1)
    X = upConv_block(X, 256, stage = 6, block=2)
    X = upConv_block(X, 128, stage = 6, block=3)
    X = upConv_block(X, 64, stage = 6, block=4)
    X = upConv_block(X, 32, stage = 6, block=5)
    
    #stage 7
    X = Conv2D(1, (3, 3), name = 'conv7', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    # Create model
    model = Model(inputs = base_model.input, outputs = X, name='BaseResNet50_preTrained_FCRN')
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def main():
    
    dataDir = 'D:/SYNS_Dataset_ForDNN3/'
    noEpochs = 50000
    batch_size = 22
    EarlyStopping_patience = 20
    
    #load pretrained resnet50 backbone
    model = ResNet50_Pretrained(input_shape = (300, 383, 3))
    
    #load Resnet50 with dropout
    #model = ResNet50_manual(input_shape = (300, 383, 3), dropoutRate = 0.4)
    
    model = multi_gpu_model(model, gpus=2)
    
    # optimizer
    optimizer_adam = tf.keras.optimizers.Adam(lr=0.00001, decay=0.01)
#    optimizer_Nadam= tf.keras.optimizers.Nadam()
    
    
    model.compile(optimizer=optimizer_adam, loss=reverseHuber, metrics=[metrics.mean_absolute_error, reverseHuber, coeff_determination])
    
    print(model.metrics_names)
    
    #tp_X = np.random.randn(5,300,383,3)
    #tp_Y = np.random.randn(5,194,258,3)
    #y_pred = model.evaluate(tp_X,tp_Y)
    
    
    # load SYNS data (train, val and test)
    pickle_in = open(dataDir + "train_X.pickle","rb")
    X_train = pickle.load(pickle_in)
    
    pickle_in = open(dataDir + "train_Y.pickle","rb")
    Y_train = pickle.load(pickle_in)
    Y_train = np.expand_dims(Y_train, axis=3) 
    
    pickle_in = open(dataDir + "test_X.pickle","rb")
    X_test = pickle.load(pickle_in)
    
    pickle_in = open(dataDir + "test_Y.pickle","rb")
    Y_test = pickle.load(pickle_in)
    Y_test = np.expand_dims(Y_test, axis=3) 
    
    pickle_in = open(dataDir + "val_X.pickle","rb")
    X_val = pickle.load(pickle_in)
    
    pickle_in = open(dataDir + "val_Y.pickle","rb")
    Y_val = pickle.load(pickle_in)
    Y_val = np.expand_dims(Y_val, axis=3) 
    
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    
    # Save the checkpoint in the /output folder
    filepath = "output/FCRN-SYNS_FlipAug_Resnet_Pretrained_optimizer_adam3-best.hdf5"
    
    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(filepath,
                                monitor='val_mean_absolute_error',
                                verbose=1,
                                save_best_only=True,
                                mode='min')
    
    preds = model.evaluate(X_val, Y_val)
    preds = model.evaluate(X_test, Y_test)
    
    # early stopping
    es = EarlyStopping(monitor='val_mean_absolute_error', 
                       mode='min', verbose=1, patience=EarlyStopping_patience)
    
    #model.fit(X_train, Y_train, epochs = 60, batch_size = 22, verbose=1, callbacks=[tensorboard, checkpoint])
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),  epochs = noEpochs, batch_size = batch_size, shuffle=True, verbose=1, callbacks=[tensorboard, checkpoint, es])
    
    #evaluate on test set
    preds = model.evaluate(X_test, Y_test)

#    
#    noEpochs = 50000
#    batch_size = 22
#    noTrainingExamples = X_train.shape[0]
#    for e in range(noEpochs):
#        print('Epoch', e)
#    
#        X_train_tmp, Y_train_tmp = shuffleData(X_train,Y_train)
#        
#        X_train_tmp[0:int(noTrainingExamples/2),:,:,:] = np.flip(X_train_tmp[0:int(noTrainingExamples/2),:,:,:],2)
#        Y_train_tmp[0:int(noTrainingExamples/2),:,:,:] = np.flip(Y_train_tmp[0:int(noTrainingExamples/2),:,:,:],2)
#        
#        X_train_tmp, Y_train_tmp = shuffleData(X_train_tmp,Y_train_tmp)
#        
#        model.fit(X_train_tmp, Y_train_tmp, validation_data=(X_val, Y_val), initial_epoch = e, epochs = e+1, batch_size = batch_size, verbose=1, callbacks=[tensorboard, checkpoint, es])
#        
#        if e%5==0:
#            preds = model.evaluate(X_test, Y_test)
    
    #tp_X = np.random.randn(5,300,383,3)
    #tp_Y = np.random.randn(5,194,258,3)
    #y_pred = model.evaluate(tp_X,tp_Y)
        
    #preds = model.evaluate(X_test, Y_test)
    #print ("Loss = " + str(preds[0]))
    #print ("Test Accuracy = " + str(preds[1]))

if __name__== "__main__":
  main()