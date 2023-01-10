import keras.backend as K
import tensorflow as tf
import math
import numpy as np

def piorHoursData(hour,dataframe):
    new_dataframe=dataframe.loc[dataframe.LOS>=hour,:]
    return new_dataframe

def extractPhysiologicalMeasurement(list_hadm_id,dataframe):
    new_dataframe=dataframe.loc[dataframe.HADM_ID.isin(list_hadm_id)]
    return new_dataframe

def imputationMean(X,means,bound):
    X_imputed=[]
    for k in range(X.shape[0]):
        all_=[]
        for j in range(X.shape[1]):
            tempo=[]
            for l in range(X.shape[2]):
                if math.isnan(X[k][j][l]):
                    tempo.append(means[l] )
#                     tempo.append(0)
                else:
                    tempo.append(X[k][j][l])
#                     tempo.append(0)
            all_.append(tempo)
        X_imputed.append(all_)
    return np.array(X_imputed).reshape(-1,bound,12) 


def intervalTimeMatrixBuild(Time,Mask,bound):
    Delta_time=[]
    for k in range(Time.shape[0]):
        all_=[]
        for j in range(Time.shape[1]):
            tempo=[]
            for l in range(Time.shape[2]):
                if j==0:
                    tempo.append(0)
                else:
                    if Mask[k][j][l]==0:
                        tempo.append(Time[k][j][l]-Time[k][j-1][l]+all_[j-1][l])
                    else:
                        tempo.append(Time[k][j][l]-Time[k][j-1][l])
            all_.append(tempo)
        Delta_time.append(all_)
    return np.array(Delta_time).reshape(-1,bound,12)
    
# Focal loss
def binary_focal_loss(gamma=2., alpha=.15):

    def binary_focal_loss_fixed(y_true, y_pred):
       
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed
