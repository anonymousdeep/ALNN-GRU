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
from collections import Counter
from utils import imputationMean,intervalTimeMatrixBuild
from loss_functions import binary_focal_loss
seed=0


# ### Data Parameters

# In[4]:


### bound=60 for the prior 24 hours and 120 for the 48 prior hours
prior_hours=24
bound=60


# ### Data

# In[5]:


Time_=pd.read_csv(f'data/time_{prior_hours}_padded_imputed.csv')
M_=pd.read_csv(f'data/mask_{prior_hours}_padded_imputed.csv')
X_=pd.read_csv(f'data/values_{prior_hours}_padded_imputed.csv')
y=pd.read_csv(f'data/target_{prior_hours}_padded_imputed.csv')


# In[6]:


#Convert data to numpy array
Time_=np.array(Time_).reshape(-1,bound,12)
M_=np.array(M_).reshape(-1,bound,12)
X_=np.array(X_).reshape(-1,bound,12)
y=y.values.reshape(-1,2)


# ### Imputation

# In[7]:


#Calculate the mean of each variable
X_tempo=np.nan_to_num(X_,nan=0)
X_tempo=X_tempo*M_
means=(np.sum(np.sum(X_tempo,axis=0),0)/np.sum(np.sum(M_,axis=0),0))
#Replace CRR missing values by the mode.
means[9]=1
del X_tempo
print(means)


# In[8]:


#Imput missing value by their corresponding mean
X_imputed=imputationMean(X_,means,bound)


# ### Create interval time matrix (Delta)

# In[9]:


Delta=intervalTimeMatrixBuild(Time_,M_,bound)


# In[11]:


#Check shapes
X_imputed.shape,Delta.shape,Time_.shape,M_.shape,y.shape


# ### Training parameters

# In[13]:


kfold = KFold(n_splits=5, shuffle=True)
bc=keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=2, mode='min')
epoch=35
start= tm.perf_counter()
MAEs,aucs,aucpr=[],[],[]


# ### Training

# In[18]:


MAEs,aucs,aucpr=[],[],[]
for train, test in kfold.split(X_imputed,y[:,0]):
    model=ALNN_GRU(24,0,25,"abs")
    model.compile(loss=binary_focal_loss(2.3,.1),optimizer=opt,metrics=["accuracy"],)

    model.fit([X_imputed[train],Time_[train],M_[train],Delta[train]], y[train][:,0],verbose=0,batch_size=100,epochs=epoch)
    loss_test, accuracy_test = model.evaluate([X_imputed[test],Time_[test],M_[test],Delta[test]],y[test][:,0],verbose=1,batch_size=100,callbacks=[callback])

    y_probas = model.predict([X_imputed[test],Time_[test],M_[test],Delta[test]]).ravel()
    fpr,tpr,thresholds=roc_curve(y[test][:,0],y_probas)
    aucs.append(auc(fpr,tpr))
    print('AUC',auc(fpr,tpr))
        
    auprc_ = sklearn.metrics.average_precision_score(y[test][:,0], y_probas)
    aucpr.append(auprc_)
    print('AUPRC', auprc_)
               
        
finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")
print(f'AUC: mean{np.round(np.mean(np.array(aucs)),3)},std{np.round(np.std(np.array(aucs)),3)}')
print(f'AUPRC: mean{np.round(np.mean(np.array(aucpr)),3)},std{np.round(np.std(np.array(aucpr)),3)}')


# ### Save model

# In[26]:


path="data/"
model.save(path)

