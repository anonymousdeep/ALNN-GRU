#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
import tensorflow as tf
import keras
# from tensorflow.keras import layers
from alnn import ALNN_GRU
import time as tm
import keras.backend as K
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import sklearn.metrics
from loss_functions import binary_focal_loss
seed=0


# ### Data Parameters

# In[ ]:


### bound=60 for the prior 24 hours and 120 for the 48 prior hours
prior_hours=24
bound=60


# ### Load data

# In[ ]:


X=pd.read_csv(f'data/Values.csv')
M=pd.read_csv(f'data/Masks.csv')
T=pd.read_csv(f'data/Time.csv')
D=pd.read_csv(f'data/Delta.csv')
y=pd.read_csv(f'data/Targets.csv')


# ### Reshape 

# In[ ]:


X=np.array(X).reshape(-1,bound,12)
M=np.array(M).reshape(-1,bound,12)
T=np.array(T).reshape(-1,bound,12)
D=np.array(D).reshape(-1,bound,12)
y=y.values


# ### Check shape

# In[ ]:


X.shape,M.shape,T.shape,D.shape,y.shape


# ### Hyperparameters

# In[ ]:


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
path="data/"


# ### Load weights and compile the model

# In[ ]:


model=ALNN_GRU(prior_hours,0,(prior_hours+1),"abs")
model.compile(loss=binary_focal_loss(2.3,.1),optimizer=opt,metrics=["accuracy"],)
model.load_weights(path)


# ### Testing

# In[ ]:


loss_test, accuracy_test = model.evaluate([X,T,M,D],y,verbose=1,batch_size=100)

y_probas = model.predict([X,T,M,D]).ravel()
fpr,tpr,thresholds=roc_curve(y,y_probas)
print('AUC',auc(fpr,tpr))

auprc_ = sklearn.metrics.average_precision_score(y, y_probas)
print('AUPRC', auprc_)

