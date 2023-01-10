#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import copy
import math
from utils import piorHoursData,extractPhysiologicalMeasurement


# In[30]:


#Define the path where the file is located
path='../../../../../data/mimic3/'


# In[31]:


#Load ICU csv file from MIMIC-3 database to extract admissions based one prior hours parameters
icu_stays=pd.read_csv(path+'ICUSTAYS.csv',low_memory=False,engine='c')
icu_stays.head(3)


# In[32]:


#1 for 24h, 2 for 48 hours
prior_hours=2
hadm_ids=piorHoursData(prior_hours,icu_stays)['HADM_ID'].unique()
print(f'There are {len(hadm_ids)} admissions for the prior {prior_hours*24} hours data.')


# In[33]:


#Load chartevents_sorted and outputsevents_sorted csv files
chartevents=pd.read_csv(path+'charevents_extracted.csv',low_memory=False,engine='c')
chartevents.head(3)


# In[34]:


#Load outputevents csv file
outputevents=pd.read_csv(path+'outputevents_extracted.csv',low_memory=False,engine='c')
outputevents.head(3)


# In[35]:


#Select measurements based on hadm_ids
chartevents=extractPhysiologicalMeasurement(hadm_ids,chartevents)
outputevents=extractPhysiologicalMeasurement(hadm_ids,outputevents)


# ## Merge features ID and Remove outliers

# <p>CRR</p>

# In[36]:


#Merge CRR itemids
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 3348 if (x==115 or x==223951 or x==8377 or x==224308)  else x)
chartevents.loc[chartevents['ITEMID']==3348,:]['VALUE'].unique()


# In[37]:


#Categorization
crr_categorical={}
crr_categorical['Brisk']=1
crr_categorical['Normal <3 secs']=1
crr_categorical['Normal <3 Seconds']=1

crr_categorical['Delayed']=2
crr_categorical['Abnormal >3 secs']=2
crr_categorical['Abnormal >3 Seconds']=2


#Change string value CRR to categorical
tempo=[]
crr_dataframe=copy.deepcopy(chartevents)
crr_dataframe=chartevents.loc[chartevents['ITEMID']==3348,:]
for item,value in zip(crr_dataframe['ITEMID'].values,crr_dataframe['VALUE'].values):
    if item==3348:
        if value not in crr_categorical:
            tempo.append(math.nan)
        else:
            tempo.append(crr_categorical[str(value)])
    else:
        tempo.append(value)
crr_dataframe=crr_dataframe.assign(VALUE=tempo)
del tempo
crr_dataframe.loc[crr_dataframe['ITEMID']==3348,:]['VALUE'].unique()


# <p>SpO2</p>

# In[38]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==220277:
        if valuenum > float(110):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary

#Merge SpO2 values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220277 if (x==646)  else x)


# <p>HR</p>

# In[39]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==220045:
        if (valuenum > float(303)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
#Merge HR values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220045 if (x==211)  else x)


# <p>RR</p>

# In[40]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==220210:
        if (valuenum > float(130)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220210 if (x==618)  else x)
#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==224690:
        if (valuenum > float(123)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
#Merge RR values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 224690 if (x==220210 or x==615 or x==618)  else x)


# <p>pH</p>

# In[41]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if (itemid_==223830) or (itemid_==4753) or (itemid_==1673) or (itemid_==1126) or (itemid_==780):
        if (valuenum > float(9)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
#Merge pH values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 780 if (x==860 or x==1126 or x==1673 or x==3839 or x==4202 or x==4753 or x==6003 or x==220274 or x==220734 or x==223830) else x)


# <p>SPB</p>

# In[42]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==220179:
        if (valuenum > float(300)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220179 if (x==455) else x)
#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==220050:
        if (valuenum > float(300)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220050 if (x==442 or x==6701 or x==51) else x)
#Merge SPB values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220050 if (x==220179) else x)


# <p>DBP</p>

# In[43]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==220180:
        if (valuenum > float(285)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220180 if (x==8441) else x)
#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==220051:
        if (valuenum > float(298)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220051 if (x==8440 or x==8555 or x==8368) else x)
#Merge DPB values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 220051 if (x==220180) else x)


# <p>Glucose</p>

# In[44]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if (itemid_==811) or (itemid_==807) or (itemid_==1529) or (itemid_==3744) or (itemid_==225664) or (itemid_==220621) or (itemid_==226537):
        if (valuenum > float(182)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)
        
chartevents['VALUENUM']=temporary
del temporary
#Merge Glucose values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 226537 if (x==807 or x==811 or x==1529 or x==3745 or x==3744 or x==225664 or x==220621)  else x)


# <p>FiO2</p>

# In[45]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==223835:
        if (valuenum > float(100)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)

chartevents['VALUENUM']=temporary
del temporary
#Merge FiO2 values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 223835 if (x==2981 or x==3420 or x==3422)  else x)


# <p>TGCS</p>

# In[46]:


#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if itemid_==223835:
        if (valuenum > float(100)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)

chartevents['VALUENUM']=temporary
del temporary
#Merge TGCS values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 198 if (x==226755)  else x)


# <p>Temp</p>

# In[47]:


#Convet Fahrenheit to Celsius and impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if (itemid_==223761) or (itemid_==678):
        if (valuenum > float(109)) or (valuenum < float(0)):
            temporary.append((valuenum-32)*5/9)
        else:
            if ((valuenum-32)*5/9)>=0:
                temporary.append((valuenum-32)*5/9)
            else:
                temporary.append(0)

    else:
        temporary.append(valuenum)

chartevents['VALUENUM']=temporary
del temporary
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 223761 if (x==678)  else x)
#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(chartevents['ITEMID'].values,chartevents['VALUENUM'].values):
    if (itemid_==223762):
        if (valuenum > float(46.5)) or (valuenum < float(0)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)

chartevents['VALUENUM']=temporary
del temporary
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 223762 if (x==676)  else x)
#Merge Temp values
chartevents['ITEMID']=chartevents['ITEMID'].apply(lambda x: 223762 if (x==223761)  else x)


# <p>UO</p>

# In[48]:


# Merge UO features
outputevents['ITEMID']=outputevents['ITEMID'].apply(lambda x: 43647 if (x == 40428 or x == 41857 or x == 42001 or x == 42362 or x == 42676 or x == 43171 or x == 43173 or x == 42042 or x == 42068 or x == 42111 or x == 42119 or x == 40715 or x == 40056 or x == 40061 or x == 40085 or x == 40094 or x == 40096 or x == 43897 or x == 43931 or x == 43966 or x == 44080 or x == 44103 or x == 44132 or x == 44237 or x == 43348 or x == 43355 or x == 43365 or x == 43372 or x == 43373 or x == 43374 or x == 43379 or x == 43380 or x == 43431 or x == 43462 or x == 43522 or x == 44706 or x == 44911 or x == 44925 or x == 42810 or x == 42859 or x == 43093 or x == 44325 or x == 44506 or x == 43856 or x == 45304 or x == 46532 or x == 46578 or x == 46658 or x == 46748 or x == 40651 or x == 40055 or x == 40057 or x == 40065 or x == 40069 or x == 44752 or x == 44824 or x == 44837 or x == 43576 or x == 43589 or x == 43633 or x == 43811 or x == 43812 or x == 46177 or x == 46727 or x == 46804 or x == 43987 or x == 44051 or x == 44253 or x == 44278 or x == 46180 or x == 45804 or x == 45841 or x == 45927 or x == 42592 or x == 42666 or x == 42765 or x == 42892 or x == 43053 or x == 43057 or x == 42130 or x == 41922 or x == 40473 or x == 43333 or x == 43347 or x == 44684 or x == 44834 or x == 43638 or x == 43654 or x == 43519 or x == 43537 or x == 42366 or x == 45991 or x == 43583)  else x)
outputevents.head(2)
#Impute outliers with nan
temporary=[]
for itemid_,valuenum in zip(outputevents['ITEMID'].values,outputevents['VALUE'].values):
    if (itemid_==43647):
        if (valuenum > float(200)):
            temporary.append(math.nan)
        else:
            temporary.append(valuenum)
    else:
        temporary.append(valuenum)

outputevents['VALUE']=temporary
del temporary

outputevents.rename(columns={'VALUE': 'VALUENUM'},inplace=True)
outputevents.head(2)


# In[49]:


#Select only columns neeeded
chartevents=chartevents[['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','VALUENUM','WARNING']]
chartevents.head(2)


# In[50]:


#Merge chartevents and outputevents dataframes
chartevents=pd.concat([chartevents,outputevents],axis=0)
del outputevents
chartevents.head(2)


# In[51]:


features=chartevents.ITEMID.unique().tolist()
features.remove(227013)
chartevents=chartevents.loc[chartevents.ITEMID.isin(features)]
print(f'Number of physiological features: {len(chartevents.ITEMID.unique())}')


# In[52]:


#Remove CRR data
chartevents = chartevents[chartevents.ITEMID !=3348]
print(f'Number of physiological features: {len(chartevents.ITEMID.unique())}')


# In[53]:


#Merge CRR dataframe and chartevents
crr_dataframe=crr_dataframe[['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','VALUE','WARNING']]
crr_dataframe.rename(columns={'VALUE': 'VALUENUM'},inplace=True)
crr_dataframe.head(2)


# In[54]:


chartevents=pd.concat([chartevents,crr_dataframe],axis=0)
del crr_dataframe
chartevents.head(2)


# In[55]:


print(f'Number of physiological features: {len(chartevents.ITEMID.unique())}')


# In[56]:


#Save the new chartevent csv file
chartevents.to_csv(f'{path}chartevents_{prior_hours*24}.csv',index=False)

