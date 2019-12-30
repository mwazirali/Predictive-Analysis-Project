# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:27:48 2019

@author: franc
"""
#import os
#get current directory
#cwd = os.getcwd()
#print(cwd)
#os.chdir('C:/Users/franc/Documents/SEM III UHD/Predictive Analytics/PYTHON/')

#----------------------------Get Data_augemented dataset----------

dict_characters = {1:'NEUTROPHIL',2:'EOSINOPHIL',3:'MONOCYTE',4:'LYMPHOCYTE'}

import numpy as np
import cv2
import scipy

import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
import os
import h5py

from keras.applications.resnet50 import ResNet50

from subprocess import check_output

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
def get_data(folder):
    X = []
    y = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 1                
            elif wbc_type in ['EOSINOPHIL']:
                label = 2                
            elif wbc_type in ['MONOCYTE']:
                label = 3                  
            elif wbc_type in ['LYMPHOCYTE']:
                label = 4                 
            else:
                label = 5                
            for image_filename in tqdm(os.listdir(folder + wbc_type)):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file,(224,224))
                    img_arr= np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)                    
    X = np.asarray(X)
    y = np.asarray(y)    
    return X,y

X_train, y_train = get_data('dataset2_master/images/TRAIN/')
X_test, y_test = get_data('dataset2_master/images/TEST/')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#SAVING TRAINING AND TESTING DATA IN h5 FILE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
hf = h5py.File("WBC_data.h5", 'w')
hf.create_dataset('x_train', data=X_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('x_test', data=X_test)
hf.create_dataset('y_test', data=y_test)
hf.close()

filename = 'WBC_data.h5'
f = h5py.File(filename, 'r')
list(f)

x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
y_test = np.array(f['y_test'])
y_train = np.array(f['y_train'])

#Normalize between 0 and 1 
x_train = x_train/255
x_test = x_test/255

classes, count = np.unique(y_train,return_counts=True)
print(classes.shape)#how many classes
print(classes) # unique class labels
print(count) # count of each class

#x_test[2,:,:,1]

print('training size', x_train.shape)
print('testing size', x_test.shape)
print('training class label size', y_train.shape)
print('testing class label size', y_test.shape)


# Encode labels to hot vectors. 
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = 5)
y_testHot = to_categorical(y_test, num_classes = 5)
#y_trainHot[:,0]
#classes, count = np.unique(y_trainHot[:,4],return_counts=True)
#print(classes.shape)#how many classes
#print(classes) # unique class labels
#print(count) # count of each class

import numpy as np
import pandas as pd

#SHOWING THE LACK OF IMBALANCE PROBLEM IN THE DATASET >>>>>>>>>>>>>>>>>>>>>
from collections import Counter
r = Counter(y_train)
print(r)

g = Counter(y_test)
print(g)

import seaborn as sns
df = pd.DataFrame()
df["labels"]=y_train
lab = df['labels']
dist = lab.value_counts()
sns.countplot(lab)
print(dict_characters)


#ACCURACY AND PLOTTING HISTORY FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#Plot the loss and accuracy during optimization process
import matplotlib
def plotHistory(Tuning):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(Tuning.history['loss'])
    axs[0].plot(Tuning.history['val_loss'])
    axs[0].set_title('loss vs epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'vali'], loc='upper left')
    
    axs[1].plot(Tuning.history['accuracy'])
    axs[1].plot(Tuning.history['val_accuracy'])
    axs[1].set_title('accuracy vs epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylim([0.0,1.0])
    axs[1].legend(['train', 'vali'], loc='upper left')
    plt.show(block = False)
    plt.show()
    
#PRINT TESTING ACCURACY
def GetAccuracy(model,predictors,response):
    from sklearn.metrics import accuracy_score
    
    #convert categorical to integer class labels
    y_classes = [np.argmax(y, axis=None, out=None) for y in response]
    pred_class = model.predict_classes(predictors)
    print('accuracy',accuracy_score(y_classes,pred_class))    

#MODEL 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image 

LOSS = 'categorical_crossentropy'
VALIDATION_SPLIT = 0.3
BATCH_SIZE = 128
EPOCHS = 50
VERBOSE = 1

modelwbc = Sequential()
#modelwbc.add(Conv2D(80, kernel_size=(3,3), activation = 'relu',padding="valid", kernel_initializer="glorot_uniform"))
modelwbc.add(Conv2D(64, (3,3), padding="valid", kernel_initializer="glorot_uniform", activation = 'relu', input_shape = (80, 80, 3)))
modelwbc.add(MaxPool2D(pool_size = (2,2)))
modelwbc.add(Conv2D(64, (3,3), activation = 'relu', padding="valid", kernel_initializer="glorot_uniform"))
modelwbc.add(BatchNormalization())
modelwbc.add(Dropout(0.25))
modelwbc.add(Flatten())

modelwbc.add(Dense(128, activation = 'relu'))
modelwbc.add(BatchNormalization())
modelwbc.add(Dropout(0.5))
modelwbc.add(Dense(5, activation = 'softmax'))

modelwbc.compile(loss = LOSS, optimizer = 'adadelta', metrics = ['accuracy'])

modelwbc.summary()

generator = image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

filepath = 'wbc_best1_2.hdf5'
checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)

history = modelwbc.fit_generator(generator.flow(x_train,y_trainHot, batch_size=32),steps_per_epoch=len(x_train) / 32, epochs= EPOCHS, validation_data = [x_test, y_testHot],callbacks = [checkpoint, early_stopping_monitor])
modelwbc.save('wbc_mod1.h5')

#Tuning_model_WBC = modelwbc.fit(x_train, y_trainHot, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=[checkpoint, early_stopping_monitor])
    
plotHistory(history)
      
print("testing accuracy for cnn model1")
GetAccuracy(modelwbc,x_test,y_testHot)

#MODEL 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from keras import optimizers
from keras.optimizers import Adam

LOSS = 'categorical_crossentropy'
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 50
EPOCHS = 30
VERBOSE = 1

modelwbc2 = Sequential()
modelwbc2.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape = (80, 80, 3)))
modelwbc2.add(Conv2D(64, (3,3), padding="valid", activation = 'relu'))
modelwbc2.add(MaxPool2D(pool_size = (2,2)))
#modelwbc2.add(Conv2D(64, (3,3), activation = 'relu', padding="valid", kernel_initializer="glorot_uniform"))
modelwbc2.add(Dropout(0.25))
modelwbc2.add(Flatten())
modelwbc2.add(Dense(128, activation = 'relu'))
modelwbc2.add(Dropout(0.5))
modelwbc2.add(Dense(5, activation = 'softmax'))
modelwbc2.compile(loss = LOSS, optimizer = 'adadelta', metrics = ['accuracy'])

modelwbc2.summary()

generator2 = image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

filepath = 'wbc_best2_2.hdf5'
checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)

history2 = modelwbc2.fit_generator(generator2.flow(x_train,y_trainHot, batch_size=32),steps_per_epoch=len(x_train) / 32, epochs= EPOCHS, validation_data = [x_test, y_testHot],callbacks = [checkpoint, early_stopping_monitor])
modelwbc2.save('wbc_mod2.h5')

#Tuning_model_WBC_2 = modelwbc2.fit(x_train, y_trainHot, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=[checkpoint, early_stopping_monitor])
    
plotHistory(history2)
GetAccuracy(modelwbc2,x_test,y_testHot)


#HYPER PARAMETERS OPTIMIZATION MODEL 2 >>>>>>>>>>>>>>>>>>>>


#STEP 1. LOAD THE DATA 

import h5py
import numpy as np
import pandas as pd

filename = 'WBC_data.h5'
f = h5py.File(filename, 'r')
list(f)

x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
y_test = np.array(f['y_test'])
y_train = np.array(f['y_train'])

#Normalize between 0 and 1 
x_train = x_train/255
x_test = x_test/255

print('training size', x_train.shape)
print('testing size', x_test.shape)
print('training class label size', y_train.shape)
print('testing class label size', y_test.shape)


#STEP 2. GRID OF PARAMETERS TO BE TUNED
np.random.seed(1000)
from keras.optimizers import SGD, Adam, Adadelta

hyperP ={'learning_rate': [1, 5, 10],
             'num_kernels': [16,32,64],
             'batch_size':[50, 100],
             'drop_out': [0.25, 0.5],
             'optimizer':[Adam, Adadelta],
             'activation':['relu'],
             'kernel_initializer':['glorot_uniform', 'glorot_normal'],
             'cnn_layers':[0,1,2,3]
             }

#STEP 3. MODEL

# Encode labels to hot vectors. 
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = 5)
y_testHot = to_categorical(y_test, num_classes = 5)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, Activation
from keras import optimizers

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from talos.model.normalizers import lr_normalizer

def cnn_model(x_train, y_trainHot, x_val, y_val, hyperP):
    INPUT_SHAPE= (80,80,3)
    num_kernels = hyperP['num_kernels']
    batch_size = hyperP['batch_size']
    cnn_layers =hyperP['cnn_layers']
    drop_out = hyperP['drop_out']
    optimizer = hyperP['optimizer']
    learning_rate = hyperP['learning_rate']
    OPTIMIZER= optimizer(lr=lr_normalizer(learning_rate, optimizer))
    
    kernel_initializer = hyperP['kernel_initializer']
    activation = hyperP['activation']
    
    model= Sequential()
    model.add(Conv2D(num_kernels, kernel_size=(3,3), kernel_initializer= kernel_initializer, input_shape=INPUT_SHAPE, data_format='channels_last'))
    model.add(Activation(activation))
    for i in range(0, cnn_layers):
        model.add(Conv2D(num_kernels, kernel_size=(3,3), kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
                   
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(drop_out))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(5, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = OPTIMIZER, metrics = ['accuracy'])
    
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
    out= model.fit(x_train, y_trainHot, batch_size=batch_size, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping_monitor])
    return out, model

#STEP 4. SCANNING THE HYPER PARAMETER SPACE
import talos

#Random reduction
t_random = talos.Scan(x=x_train, y=y_trainHot,
                      model= cnn_model, 
                      dataset_name='wbc',
                      experiment_no = '2',
                      params= hyperP,
                      grid_downsample= 0.5)

talos.Deploy(t_random, "talos_random4_tuning", metric='accuracy')

#try: 
 #   talos.Deploy(t_random, "talos_random3_tuning", metric='val_acc')
#except:
 #   print(" ")


def OutputResults(x_data,y_data,bestmodel_json_file,bestmodel_weights_h5_file):
    from keras.models import model_from_json
    file = open(bestmodel_json_file,'r')
    lines = file.read()
    file.close()
    model = model_from_json(lines)
    model.load_weights(bestmodel_weights_h5_file)
    print(model.summary())
    pred_class = model.predict_classes(x_data)
    from sklearn.metrics import accuracy_score
    print('testing accuracy',accuracy_score(y_data,pred_class))


OutputResults(x_test, y_test,'talos_random4_tuning/talos_random4_tuning_model.json','talos_random4_tuning/talos_random4_tuning_model.h5')


#HYPERPARAMETER OPTIMIZATION MODEL 1 >>>>>>>>>>>>>>>>>>>>>>>
np.random.seed(1003)
from keras.optimizers import SGD, Adam, Adadelta

hyperP2 ={'learning_rate': [1, 10],
             'num_kernels': [32,64],
             'batch_size':[50, 100],
             'drop_out': [0.25, 0.5],
             'optimizer':[Adam, Adadelta],
             'activation':['relu'],
             'kernel_initializer':['glorot_uniform', 'glorot_normal'],
             'cnn_layers':[0,1,2,3]
             }

from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image 

def cnn_model2(x_train, y_trainHot, x_val, y_val, hyperP):
    INPUT_SHAPE= (80,80,3)
    num_kernels = hyperP['num_kernels']
    batch_size = hyperP['batch_size']
    cnn_layers =hyperP['cnn_layers']
    drop_out = hyperP['drop_out']
    optimizer = hyperP['optimizer']
    learning_rate = hyperP['learning_rate']
    OPTIMIZER= optimizer(lr=lr_normalizer(learning_rate, optimizer))
    
    kernel_initializer = hyperP['kernel_initializer']
    activation = hyperP['activation']
    
    model= Sequential()
    model.add(Conv2D(num_kernels, kernel_size=(3,3), kernel_initializer= kernel_initializer, input_shape=INPUT_SHAPE, data_format='channels_last'))
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2,2)))
    #model.add(Conv2D(num_kernels, kernel_size=(3,3), kernel_initializer= kernel_initializer, input_shape=INPUT_SHAPE, data_format='channels_last'))
    #model.add(Activation(activation))
    for i in range(0, cnn_layers):
        model.add(Conv2D(num_kernels, kernel_size=(3,3), kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
                   
    model.add(BatchNormalization())
    model.add(Dropout(drop_out))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out))
    model.add(Dense(5, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = OPTIMIZER, metrics = ['accuracy'])
    
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
    out= model.fit(x_train, y_trainHot, batch_size=batch_size, epochs=10, verbose=0, validation_split=0.2, callbacks=[early_stopping_monitor])
    return out, model

t_random3 = talos.Scan(x=x_train, y=y_trainHot,
                      model= cnn_model2, 
                      dataset_name='wbc_2',
                      experiment_no = '3',
                      params= hyperP2,
                      grid_downsample= 0.1)


talos.Deploy(t_random3, "talos_random_tuning_model3", metric='accuracy')


OutputResults(x_test, y_test,'talos_random_tuning_model2/talos_random_tuning_model2_model.json','talos_random_tuning_model2/talos_random_tuning_model2_model.h5')



#RANDOM FOREST >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
filename = 'WBC_data.h5'
f = h5py.File(filename, 'r')
list(f)

x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
y_test = np.array(f['y_test'])
y_train = np.array(f['y_train'])

f.close()

#HOG for train dataset--------------------------------------------------------------
x_train_hog = np.empty(shape=(x_train.shape[0],200))
for i in range(0,x_train.shape[0]):
    fd = hog(x_train[0,:,:], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)
    x_train_hog[i,:] = fd

x_train_hog.shape


#HOG for test dataset-----------------------------------------------------------------
x_test_hog = np.empty(shape=(x_test.shape[0],200))
for i in range(0,x_test.shape[0]):
    fd = hog(x_test[0,:,:], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, multichannel=True, block_norm= 'L2')
    x_test_hog[i,:] = fd

x_test_hog.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#Test Random Forest Model
#Z, t = make_classification(n_samples=1000, n_features=5,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
#clf2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0) 
#model2 = clf2.fit(Z,t)
#print(clf2.predict([[ 0, -0.4, -0.5, -0.5,  1]]))
#[0]

clf2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
model2 = clf2.fit(x_train_hog,y_train)
preds2=clf.predict(x_test_hog)

print(confusion_matrix(y_test, preds2))
print(classification_report(y_test, preds2))


#KNN - for testing purpose---------------------------------------------------------------------
#scaler = StandardScaler()
#scaler.fit(x_train_hog)

#train_hog = scaler.transform(x_train_hog)
#test_hog = scaler.transform(x_test_hog)

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=10)
#classifier.fit(x_train, y_train)

#y_pred = classifier.predict(test_hog)




#SVM - for testing purpose-----------------------------------------------------------------------------
#clf = svm.SVC()
#model=clf.fit(x_train_hog,y_train)
#preds =  model.predict(x_test_hog)


#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, preds))
#print(classification_report(y_test,preds))






#INCEPTIONV3-----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
# Input data files are available in the "../input/" directory.


# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import warnings
warnings.filterwarnings('ignore')
data_path= "../input/blood-cells/dataset2-master/dataset2-master/images"
# Any results you write to the current directory are saved as output.


import cv2 
import matplotlib.pyplot as plt
from keras.preprocessing.image import *
from keras.applications import InceptionV3,VGG16
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import layers, optimizers
from keras.models import *
import scipy
from keras.models import load_model

from sklearn.metrics import classification_report,accuracy_score


#os.chdir('/home/dluser9/CS6302/dataset2-master/images')
os.chdir('/home/dluser9/CS6302')
filename = 'WBC_data.h5'
f = h5py.File(filename, 'r')
list(f)

x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
y_test = np.array(f['y_test'])
y_train = np.array(f['y_train'])

#Normalize between 0 and 1 
x_train = x_train/255
x_test = x_test/255




from tqdm import tqdm
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    z = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 1
                label2 = 1
            elif wbc_type in ['EOSINOPHIL']:
                label = 2
                label2 = 1
            elif wbc_type in ['MONOCYTE']:
                label = 3  
                label2 = 0
            elif wbc_type in ['LYMPHOCYTE']:
                label = 4 
                label2 = 0
            else:
                label = 5
                label2 = 0
            for image_filename in tqdm(os.listdir(folder + wbc_type)):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
                    z.append(label2)
    X = np.asarray(X)
    y = np.asarray(y)
    z = np.asarray(z)
    return X,y,z

os.chdir('/home/dluser9/CS6302/dataset2-master/images')
X_train, y_train, z_train = get_data('TRAIN/')
X_test, y_test, z_test = get_data('TEST/')

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = 5)
y_testHot = to_categorical(y_test, num_classes = 5)
z_trainHot = to_categorical(z_train, num_classes = 2)
z_testHot = to_categorical(z_test, num_classes = 2)
dict_characters = {1:'NEUTROPHIL',2:'EOSINOPHIL',3:'MONOCYTE',4:'LYMPHOCYTE'}
dict_characters2 = {0:'Mononuclear',1:'Polynuclear'}
print(dict_characters)
print(dict_characters2)



classes, count = np.unique(y_train,return_counts=True)
print(classes.shape)#how many classes
print(classes) # unique class labels
print(count) # count of each class


def get_model():
    base_mdoel = InceptionV3(weights='imagenet',include_top=False,input_shape=img_shape)
    model= Sequential()
    model.add(base_mdoel)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='elu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(128, activation='elu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='elu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(5, activation='softmax'))
    model.summary()
    optimizer = optimizers.Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model



def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0:
        return (num_samples // batch_size) + 1
    else:
        return num_samples // batch_size


batch_size=32
model_path = '../model/'
if not os.path.exists(model_path):
    os.mkdir(model_path)


model_path = model_path + 'best_model.hdf5'
train_num = X_train.shape[0]
test_num =  X_test.shape[0]


callbacks = [EarlyStopping(monitor = 'val_loss',verbose=1, patience = 2,mode='min'), ReduceLROnPlateau(monitor = 'val_acc',verbose=1, factor = 0.5, patience = 1, min_lr=0.00001, mode='min'),
             ModelCheckpoint(filepath=model_path,verbose=1, monitor='val_loss', save_best_only=True, mode='min'),]


sample_data3_path = os.path.join('TRAIN/LYMPHOCYTE','_14_8262.jpeg')
sample_data3 = cv2.imread(sample_data3_path)
img_shape =sample_data3.shape 


my_inception_model =get_model()

history = my_inception_model.fit(
    X_train,
    y_trainHot,
    batch_size=batch_size,
    epochs=30,
    verbose=1,
    validation_data=(X_test,y_testHot),
    callbacks=callbacks)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='Traing loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Trainging and validation loss')
plt.legend()
plt.show()


model = load_model(model_path + 'best_model.hdf5')

preds = model.predict(X_test[200:500])

y_classes = preds.argmax(axis=-1)


print(classification_report(y_test, y_classes))
print('accuracy',accuracy_score(y_test,y_classes))    


hf = h5py.File("Inception_data.h5", 'w')
hf.create_dataset('preds', data=preds)
hf.create_dataset('y_classes', data=y_classes)
hf.create_dataset('x_test', data=X_test)
hf.create_dataset('y_test', data=y_test)
hf.close()

## Alex_Net Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras import backend
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report,accuracy_score


#Blood cell subtype classification
import cv2 as cv
import seaborn as sns

# for the Tensorflow backend
{"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow", "image_data_format": "channels_first"}

#TRAIN AND TEST DATASET ADDRESS

DATASET="/home/dluser23/dataset2-master/dataset2-master/images/TRAIN"
TEST_DATASET="/home/dluser23/dataset2-master/dataset2-master/images/TEST"

#Categroized images
#4 types of subcells
CATEGORIES=["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]

#reading original image from directory
for category in CATEGORIES:
        label=CATEGORIES.index(category)
        path=os.path.join(DATASET,category)
        
        for img_file in os.listdir(path):
            
            # 1 indicates read image in RGB scale
            # 0 indicates read image in grey scale
            
            img=cv.imread(os.path.join(path,img_file),1)
            
           #open cv read image in BGR format 
            #below we convert it to RGB format
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            #print(img.shape)
            plt.imshow(img)
            plt.show()
            break
            
#plotting single image from each folder
#reading image from directory
for category in CATEGORIES:
        label=CATEGORIES.index(category)
        path=os.path.join(DATASET,category)
        
        for img_file in os.listdir(path):
            
            img=cv.imread(os.path.join(path,img_file),1)
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            dst = cv.fastNlMeansDenoisingColored(img,None,5,10,7,21)
            #image convert to smaller pixels 60*60
            print(img.shape)
            plt.figure(figsize=(10,8))
            plt.subplot(121)
            plt.imshow(dst)
            plt.subplot(122)
            plt.imshow(img)
            plt.show()
            break
            
#plotting single image from each folder
            
#make train data
train_data=[]

for category in CATEGORIES:
    
        #each cateogry into unique integer
        label=CATEGORIES.index(category)
        path=os.path.join(DATASET,category)
        
        for img_file in os.listdir(path):
            
            img=cv.imread(os.path.join(path,img_file),1)
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            #dst = cv.fastNlMeansDenoisingColored(img,None,5,10,7,21)
            img=cv.resize(img,(224,224))            
            train_data.append([img,label])
            
#make test data
test_data=[]

for category in CATEGORIES:
       
        #each cateogry into unique integer
        label=CATEGORIES.index(category)
        path=os.path.join(TEST_DATASET,category)
        
        for img_file in os.listdir(path):
            
            img=cv.imread(os.path.join(path,img_file),1)
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            #dst = cv.fastNlMeansDenoisingColored(img,None,5,10,7,21)
            img=cv.resize(img,(224,224))
            test_data.append([img,label])

#print total data in train and test
#print(len(train_data))
#print(len(test_data))

#shuffle the dataset for good result

import random

random.shuffle(train_data)
random.shuffle(test_data)

#check the data
for lbl in train_data[:10]:
    print(lbl[1])
for lbl in test_data[:10]:
    print(lbl[1])
    
#lets seprate the feature and target variable in train data
train_X=[]
train_y=[]

for features,label in train_data:
    train_X.append(features)
    train_y.append(label)

len(train_X),len(train_y)

#lets seprate the feature and target variable in test data
test_X=[]
test_y=[]

for features,label in test_data:
    test_X.append(features)
    test_y.append(label)

len(test_X),len(test_y)

#convert image array to numpy array
train_X = np.array(train_X).reshape(-1,224,224,3)
train_X = train_X/255.0
train_X.shape


test_X=np.array(test_X).reshape(-1,224,224,3)
test_X=test_X/255.0
test_X.shape


#count labels

sns.countplot(train_y,palette='Set3')

#convert label into the one hot encode
from tensorflow.keras.utils import to_categorical
#train y
one_hot_train=to_categorical(train_y)
#one_hot_train

#test_y
one_hot_test=to_categorical(test_y)
#one_hot_test
  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np
np.random.seed(1000)

def get_model():
    
    model = Sequential()

# 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding= 'valid'))
    model.add(Activation('relu'))

# Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding= 'valid'))

# Batch Normalization
    model.add(BatchNormalization())

# 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding= 'valid'))
    model.add(Activation('relu'))

# Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding= 'valid'))
# Batch Normalization
    model.add(BatchNormalization())

# 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding= 'valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())


# 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),padding= 'valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())


# 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding= 'valid'))
    model.add(Activation('relu'))
# Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding= 'valid'))

# Batch Normalization
    model.add(BatchNormalization())

# Passing it to a Fully Connected layer
    model.add(Flatten())

# 1st Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
# Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
# Batch Normalization
    model.add(BatchNormalization())

# 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
# Add Dropout
    model.add(Dropout(0.4))
# Batch Normalization
    model.add(BatchNormalization())

# 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))

# Add Dropout
    model.add(Dropout(0.4))

# Batch Normalization
    model.add(BatchNormalization())

# Output Layer
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.summary()   

# Compile the model
    METRICS =['accuracy']
    #ADAM = optimizers.Adam(lr=0.0001)
    model.compile(loss= tf.keras.losses.categorical_crossentropy, optimizer= 'adam', metrics= METRICS)

    return model
     
   
batch_size= 64

filepath ='AN_best.hdf5'
train_num = train_X.shape[0]
test_num =  test_X.shape[0]
callbacks = [EarlyStopping(monitor='val_loss', mode = 'min', verbose = 1, patience = 8),
             ModelCheckpoint(filepath= 'AN_best.hdf5',verbose=1, monitor='val_loss', save_best_only=True, mode='min'),]


an_model = get_model()

history = an_model.fit(
    train_X,
    one_hot_train,
    batch_size=batch_size,
    epochs=50,
    verbose=1,
    validation_split=(0.2),
    callbacks=callbacks)
#Plot the loss and accuracy during optimization process

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
def plotHistory(Tuning):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(Tuning.history['loss'])
    axs[0].plot(Tuning.history['val_loss'])
    axs[0].set_title('loss vs epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'vali'], loc='upper left')
    plt.show(block = False)
    plt.show()
    plt.savefig('./loss_curve.png')
    
    axs[1].plot(Tuning.history['acc'])
    axs[1].plot(Tuning.history['val_acc'])
    axs[1].set_title('accuracy vs epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylim([0.0,1.0])
    axs[1].legend(['train', 'vali'], loc='upper left')
    plt.show(block = False)
    plt.show()
    plt.savefig('./accuracy_curve.png')
    


plotHistory(history)


filepath ='AN_best.hdf5'
an_best_model = load_model('AN_best.hdf5')
an_best_model.summary()

preds = an_best_model.predict_classes(test_X)

y_classes = preds.argmax(axis=-1)

#from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score,confusion_matrix
#y_classes = preds.argmax(axis=-1)
print(confusion_matrix(test_y, preds))
print(classification_report(test_y, preds))
print('Testing Accuracy',accuracy_score(test_y,preds))

# Architecture of the best model
print('Architecture of the best model:')
an_best_model.summary()

#Compress the Data so that is uploadable to the drive 
import numpy as np
import h5py
import gzip
import shutil
hf = h5py.File("AN_best.hdf5", 'r+')
hf = hf.create_dataset('dataset', (100000,), dtype='i1', compression="gzip", compression_opts=9)
filepath = 'AN_final.h5'
tf.keras.models.save_model(hf,filepath)
#with gzip.open(filepath, 'r') as f_in, open(filepath, 'r') as f_out:
#    d = shutil.copyfileobj(f_in, f_out)
#    file = d
#model_final = load_model(file)
#print(model_final.summary())
