import math
import numpy as np


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