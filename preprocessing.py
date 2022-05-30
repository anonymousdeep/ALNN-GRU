#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time as tm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math


# In[ ]:


#Load chartevent_{prior_hours}_imputed csv file processed in data_imputation.py
prior_hours=24
chartevents=pd.read_csv(f'../data/mimic3/chartevents_{prior_hours}_imputed.csv',low_memory=False,engine='c')
icustays=pd.read_csv('../data/mimic3/ICUSTAYS.csv',low_memory=False,engine='c')
admissions=pd.read_csv('../data/mimic3/ADMISSIONS.csv',low_memory=False,engine='c')
d_items=pd.read_csv('../data/mimic3/D_ITEMS.csv',engine='c',low_memory=False)


# In[ ]:


#Data description
nr_features=len(chartevents.ITEMID.unique())
nr_subjects=len(chartevents.SUBJECT_ID.unique())
nr_hadms=len(chartevents.HADM_ID.unique())
print(f'Number of features {nr_features}')
print(f'Number of subjects {nr_subjects}')
print(f'Number of hadms {nr_hadms}')


# In[ ]:


print(chartevents.ITEMID.unique())


# ### Inputs & Outputs 

# <p>Get intime of all patients</p>

# In[ ]:


ids_stay_intime={}
for s,i in zip(icustays['HADM_ID'].values,icustays['INTIME'].values):
    ids_stay_intime[s]=i


# <p>Get all HADM_ID in chartevents</p>

# In[ ]:


hadm_ids=chartevents['HADM_ID'].unique()
len(hadm_ids)


# <p>Get hadm_id and state of the patietns afetr each admission.</p>

# In[ ]:


dict_state={}
classs=[]
start= tm.perf_counter()
for idnx,(hadm,statepatient) in enumerate(zip(admissions['HADM_ID'].values,admissions['HOSPITAL_EXPIRE_FLAG'].values)):
    if hadm in hadm_ids:
        dict_state[hadm]=statepatient
        classs.append(statepatient)
    #Progress monitoring
#     if idnx%10000==0:
#         print(idnx)
finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")


# <p>Class distribution</p>

# In[ ]:


pd.DataFrame(data=classs)[0].value_counts()


# In[ ]:


chartevents=chartevents[['HADM_ID','ITEMID','CHARTTIME','VALUENUM','MASK']]
chartevents.head(2)


# In[ ]:


features=chartevents['ITEMID'].unique()


# <p> Get only the first prior hours data</p>

# In[ ]:


data={}
flags=[]
hdms=[]
tempo=[]
features=chartevents['ITEMID'].unique()
max_length=0
array_of_of_length=[]
start= tm.perf_counter()
for idnx,hadmid in enumerate(hadm_ids):   
    init_date=datetime.strptime(ids_stay_intime[hadmid], '%Y-%m-%d %H:%M:%S')
    sub_dataframe=chartevents.loc[chartevents['HADM_ID']==hadmid,:].sort_values(by="CHARTTIME")
    data[hadmid]={}
    for item in features:
        sub_dataframe_tempo=sub_dataframe.loc[sub_dataframe['ITEMID']==item,:].sort_values(by="CHARTTIME").values
        tempo=[]
#         for idx,(chartime,valuenum) in enumerate(zip(sub_dataframe_tempo[:,2],sub_dataframe_tempo[:,3])):
        for idx,(chartime,valuenum,msk) in enumerate(zip(sub_dataframe_tempo[:,2],sub_dataframe_tempo[:,3],sub_dataframe_tempo[:,4])):
            current_date=datetime.strptime(chartime, '%Y-%m-%d %H:%M:%S')
            diff = relativedelta(current_date,init_date)
            timestamp=diff.days*24 + diff.hours #+ diff.minutes/60 + diff.seconds/3600
            
            if (timestamp<=prior_hours) and (timestamp>=0):
                tempo.append([timestamp,valuenum,msk])
            elif (timestamp>prior_hours):
                break
                
#             if (timestamp<=prior_hours) and (timestamp>=0):
#                 if math.isnan(valuenum):
#                     tempo.append([timestamp,valuenum,0])
#                 else:
#                     tempo.append([timestamp,valuenum,1])
#             elif (timestamp>prior_hours):
#                 break
                
              
        if len(tempo)>0:
            data[hadmid][item]=[]
            data[hadmid][item].append(tempo)
            array_of_of_length.append(len(tempo))
            
    if idnx%10000==0:
        print(idnx)
finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")


# In[ ]:


del chartevents


# In[ ]:


# bound=60 for prior_hours=24 and bound=120 for prior_hours=24
counter=0
bound=60
for k in array_of_of_length:
    if k >bound:
        counter=counter+1  

print(f'Number of samples with lenght { bound} equals {counter}.')
print(f'Percentage of samples {((counter*100)/len(array_of_of_length))}% with size grater than {bound}.')


# In[ ]:


X,Time,M,y=[],[],[],[]
axis_0=0
bound_hadmid_less_48,bound_hadmid_48=[],[]
erro_admin=[]
for indx,hadmid in enumerate(hadm_ids):      
    axis_0=axis_0+1
    xs,ts,m=[],[],[]
    past_dic={}
    past_time={}
    for i in range(bound):
        for feature in features:
            if feature in data[hadmid]:
                if i<= len(data[hadmid][feature][0])-1:
                    m.append(data[hadmid][feature][0][i][2])
                    xs.append(data[hadmid][feature][0][i][1])
                    ts.append(data[hadmid][feature][0][i][0])
                    past_dic[feature]=data[hadmid][feature][0][i][1]
                    past_time[feature]=data[hadmid][feature][0][i][0]
                else:
                    m.append(0)
                    xs.append(past_dic[feature])
                    ts.append(past_time[feature])

            else:
                m.append(0)
                ts.append(0)
                xs.append(math.nan)
    
    
    X.append(xs)
    Time.append(ts)
    M.append(m)

    y.append([dict_state[hadmid],hadmid])
      
    if indx%10000==0:
        print(indx)


# In[ ]:


X_=np.array(X).reshape(-1,bound,12)
Time_=np.array(Time).reshape(-1,bound,12)
M_=np.array(M).reshape(-1,bound,12)
y=np.array(y).reshape(-1,2)
X_.shape,Time_.shape,M_.shape,H_.shape,R_.shape,NT_.shape,PT_.shape,y.shape


# In[ ]:


pd.DataFrame(data=Time_.reshape(-1,bound*len(features))).to_csv(f'data/time_{prior_hours}_padded_imputed.csv',index=False)
pd.DataFrame(data=M_.reshape(-1,bound*len(features))).to_csv(f'data/mask_{prior_hours}_padded_imputed.csv',index=False)
pd.DataFrame(data=X_.reshape(-1,bound*len(features))).to_csv(f'data/values_{prior_hours}_padded_imputed.csv',index=False)
pd.DataFrame(data=y.reshape(-1,2)).to_csv(f'data/target_{prior_hours}_padded_imputed.csv',index=False)


# In[ ]:




