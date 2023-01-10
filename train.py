#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import math
import tensorflow as tf
import keras
from tensorflow.keras import layers
from alnn import ALNN_GRU
from sklearn.model_selection import KFold
import time as tm
import keras.backend as K
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from utils import binary_focal_loss,imputationMean,intervalTimeMatrixBuild


# In[4]:


tf.random.set_seed(14)


# In[7]:


#Define the path where the file is located
path='../data/'


# In[6]:


prior_hours=48
stream_length=120#60 for 24 hours


# In[8]:


Time_=pd.read_csv(f'{path}time_{prior_hours}_{stream_length}_padded_imputed.csv')
M_=pd.read_csv(f'{path}mask_{prior_hours}_{stream_length}_padded_imputed.csv')
X_=pd.read_csv(f'{path}values_{prior_hours}_{stream_length}_padded_imputed.csv')
y=pd.read_csv(f'{path}target_{prior_hours}_{stream_length}_padded_imputed.csv')


# In[9]:


Time_=np.array(Time_).reshape(-1,stream_length,12)
M_=np.array(M_).reshape(-1,stream_length,12)
X_=np.array(X_).reshape(-1,stream_length,12)
y=y.values.reshape(-1)
X_.shape,Time_.shape,M_.shape,y.shape


# In[10]:


#Compute the empirical mean of each feature
X_tempo=np.nan_to_num(X_,nan=0)
X_tempo=X_tempo*M_
means=(np.sum(np.sum(X_tempo,axis=0),0)/np.sum(np.sum(M_,axis=0),0))
#Replace CRR missing values by the mode.
means[9]=1
del X_tempo
means


# In[11]:


#Imput missing value by their corresponding mean
X_imputed=imputationMean(X_,means,stream_length)


# In[12]:


# Create interval time matrix (Delta)
Delta=intervalTimeMatrixBuild(Time_,M_,stream_length)


# In[13]:


#Check shapes
print(f' Values matrix shape {X_imputed.shape}')
print(f' Masks matrix shape {M_.shape}')
print(f' TimeInterveals matrix shape {Delta.shape}')
print(f' Time matrix shape {Time_.shape}')
print(f' Target shape {y.shape}')


# In[14]:


#Training parameters
kfold = KFold(n_splits=5, shuffle=True,random_state=14)
bc=keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3, mode='min')
epoch=35
start= tm.perf_counter()
MAEs,aucs,aucpr=[],[],[]


# In[15]:


#Set number value in Delta r
nr_values=1
print(prior_hours*nr_values+1)


# In[19]:


#Training
MAEs,aucs,aucpr=[],[],[]
for train, test in kfold.split(X_imputed,y):
    model=ALNN_GRU(prior_hours,0,prior_hours*nr_values+1,"abs")
    model.compile(loss=binary_focal_loss(2.1,0.1),optimizer=opt,metrics=["accuracy"])

    model.fit(
        [X_imputed[train],Time_[train],
        M_[train],Delta[train]], y[train],
        verbose=0,batch_size=200,
        epochs=epoch)


    loss_test, accuracy_test = model.evaluate([X_imputed[test],Time_[test],M_[test],Delta[test]],y[test],verbose=1,batch_size=200)

    y_probas = model.predict([X_imputed[test],Time_[test],M_[test],Delta[test]]).ravel()
    fpr,tpr,thresholds=roc_curve(y[test],y_probas)
    aucs.append(auc(fpr,tpr))
    print('AUC',auc(fpr,tpr))

    auprc_ = sklearn.metrics.average_precision_score(y[test], y_probas)
    aucpr.append(auprc_)
    print('AUPRC', auprc_)


finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")
print(f'AUC: mean{np.round(np.mean(np.array(aucs)),3)},std{np.round(np.std(np.array(aucs)),3)}')
print(f'AUPRC: mean{np.round(np.mean(np.array(aucpr)),3)},std{np.round(np.std(np.array(aucpr)),3)}')
print('\n')

