#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time as tm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
from tqdm import tqdm


# In[ ]:


#Define the path where the file is located
path='../../../../../data/mimic3/'


# In[ ]:


#Load chartevent_{prior_hours}_imputed csv file processed in data_imputation.py
prior_hours=48
chartevents=pd.read_csv(f'{path}chartevents_{prior_hours}_imputed.csv',low_memory=False,engine='c')
icustays=pd.read_csv(f'{path}ICUSTAYS.csv',low_memory=False,engine='c')
admissions=pd.read_csv(f'{path}ADMISSIONS.csv',low_memory=False,engine='c')
d_items=pd.read_csv(f'{path}D_ITEMS.csv',engine='c',low_memory=False)


# In[ ]:


#Data description
nr_features=len(chartevents.ITEMID.unique())
nr_subjects=len(chartevents.SUBJECT_ID.unique())
nr_hadms=len(chartevents.HADM_ID.unique())
print(f'Number of features {nr_features}')
print(f'Number of subjects {nr_subjects}')
print(f'Number of hadms {nr_hadms}')


# In[ ]:


print(f'Physiological measurement IDs=> {chartevents.ITEMID.unique()}')


# ### Extraction of the inputs and outputs

# In[ ]:


#Record the INTIME to the ICU of all patients
ids_stay_intime={}
for s,i in zip(icustays['HADM_ID'].values,icustays['INTIME'].values):
    ids_stay_intime[s]=i


# In[ ]:


#Get all HADM_ID in chartevents
hadm_ids=chartevents['HADM_ID'].unique()
print(f'Number of admissions in ICU:{len(hadm_ids)}')


# In[ ]:


#Get hadm_id and state of the patietns afetr each admission.
dict_state={}
classs=[]
start= tm.perf_counter()
for idnx,(hadm,statepatient) in enumerate(zip(admissions['HADM_ID'].values,admissions['HOSPITAL_EXPIRE_FLAG'].values)):
    if hadm in hadm_ids:
        dict_state[hadm]=statepatient
        classs.append(statepatient)

finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")


# In[ ]:


#Class distribution
pd.DataFrame(data=classs)[0].value_counts()#O for died 1 for alive


# In[ ]:


#Select only the features needed
chartevents=chartevents[['HADM_ID','ITEMID','CHARTTIME','VALUENUM','MASK']]
chartevents.head(2)


# In[ ]:


features=chartevents['ITEMID'].unique()
prior_hours


# In[ ]:


#Extract physiological measurements, their corresponding timestamps and mask values
data={}
flags,hdms,array_of_of_length=[],[],[]
features=chartevents['ITEMID'].unique()
max_length=0

#tqdm is just for the progress bar
for hadmid in tqdm(hadm_ids): 
    
    #ICU INTIME of the current patient 
    init_date=datetime.strptime(ids_stay_intime[hadmid], '%Y-%m-%d %H:%M:%S')
    #All data of the patient at the current admission
    sub_dataframe=chartevents.loc[chartevents['HADM_ID']==hadmid,:].sort_values(by="CHARTTIME")
    
    data[hadmid]={}
    
    #We iterate across each selected features (streams) to save their values, timestamps and masks
    for item in features:
        sub_dataframe_tempo=sub_dataframe.loc[sub_dataframe['ITEMID']==item,:].sort_values(by="CHARTTIME").values
        #Temporary array to save timestamps, values and masks on each iteration
        tempo=[]
        
        #We iterate over the stream
        for idx,(chartime,valuenum,msk) in enumerate(zip(sub_dataframe_tempo[:,2],sub_dataframe_tempo[:,3],sub_dataframe_tempo[:,4])):
            
            #Current at which the value was recorded
            current_date=datetime.strptime(chartime, '%Y-%m-%d %H:%M:%S')
            #Subtract the current date from the intime date to obtain the timestamp value
            diff = current_date-init_date
            #Convert the timestamp in hours
            timestamp=diff.days*24 + diff.seconds/3600

            #If the timestamp is less than the prior_hours parameter, we save the timestamp, the value and the mask
            #Otherwise, we stop looping over the strea
            if (timestamp<=prior_hours) and (timestamp>=0):
                tempo.append([timestamp,valuenum,msk])
            elif (timestamp>prior_hours):
                break

        #We check if the patient has values for the current feature
        if len(tempo)>0:
            data[hadmid][item]=[]
            data[hadmid][item].append(tempo)
            array_of_of_length.append(len(tempo))#Extract physiological measurements, their corresponding timestamps and mask values


# In[ ]:


#As we no longer need it, we delete the chartevents dataframe to save memory
del chartevents


# In[ ]:


counter=0
stream_length=120#This parameter specifies how many values considered by stream
for k in array_of_of_length:
    if k >stream_length:
        counter=counter+1  

print(f'Number of samples with lenght { stream_length} equals {counter}.')
print(f'Percentage of samples with size grater than {stream_length} : {((counter*100)/len(array_of_of_length))}%.')


# In[ ]:


#Padding process
X,Time,M,y,idshadm=[],[],[],[],[]
axis_0=0

erro_admin=[]
for hadmid in tqdm(hadm_ids):      
    axis_0=axis_0+1
    xs,ts,m=[],[],[]
    past_dic={}
    past_time={}
    for i in range(stream_length):
        for feature in features:
            if feature in data[hadmid]:
                if i<= len(data[hadmid][feature][0])-1:
                    m.append(data[hadmid][feature][0][i][2])
                    xs.append(data[hadmid][feature][0][i][1])
                    ts.append(data[hadmid][feature][0][i][0])
                    past_dic[feature]=data[hadmid][feature][0][i][1]
                    past_time[feature]=data[hadmid][feature][0][i][0]
                else:
                    #Missin value
                    m.append(0)
                    #Fill forward with the last values observed
                    xs.append(past_dic[feature])
                    #Fill forward with the last timestamp observed
                    ts.append(past_time[feature])

            else:
                #Missin value
                m.append(0)
                #Fill forward with the 0 for the timestamp and nan for the value. It means that the current patient has no value for the current feature
                #This nan value will be replaced by the mean of observed values of the current feature
                #The value of timstamp here can be any value. We keep it to zero so that it has not impact in our model since it is missing value 
                ts.append(0)
                xs.append(math.nan)
    
    X.append(xs)
    Time.append(ts)
    M.append(m)
    y.append(dict_state[hadmid])
    idshadm.append(hadmid)


# In[ ]:


X=np.array(X).reshape(-1,stream_length,12)
Time=np.array(Time).reshape(-1,stream_length,12)
M=np.array(M).reshape(-1,stream_length,12)
y=np.array(y).reshape(-1,)
print(f'Shape of observed values matrix{X.shape}')
print(f'Shape of the timestamps matrix {Time.shape}')
print(f'Shape of the masks matrix {M.shape}')
print(f'Number of targeted features {y.shape}')


# In[ ]:


#Save each input and the target in datafram form.
pd.DataFrame(data=Time.reshape(-1,stream_length*len(features))).to_csv(f'{path}time_{prior_hours}_{stream_length}_padded_imputed.csv',index=False)
pd.DataFrame(data=M.reshape(-1,stream_length*len(features))).to_csv(f'{path}mask_{prior_hours}_{stream_length}_padded_imputed.csv',index=False)
pd.DataFrame(data=X.reshape(-1,stream_length*len(features))).to_csv(f'{path}values_{prior_hours}_{stream_length}_padded_imputed.csv',index=False)
pd.DataFrame(data=y.reshape(-1)).to_csv(f'{path}target_{prior_hours}_{stream_length}_padded_imputed.csv',index=False)

