import pandas as pd
import numpy as np


# In[ ]:


#Load chartevents csv file from  MIMIC-3 database
chartevents=pd.read_csv('data/CHARTEVENTS.csv',low_memory=False,engine='c')
chartevents.head(5)


# In[ ]:


#Features
SpO2=[646,220277]#Oxygen saturation levels
HR=[211, 220045]#Heart Rate
RR=[618, 615, 220210, 224690]#Respiratory Rate
SBP=[51,442,455,6701,220179,220050]#Systolic Blood Pressure
DBP=[8368,8440,8441,8555,220180,220051]#Diastolic Blood Pressure
FiO2=[2981, 3420, 3422, 223835]#Fraction inspired oxygen
Glucose=[807,811,1529,3745,3744,225664,220621,226537]
Temp_F=[223761,678]#Temperature Fahrenheit
Temp_C=[223762,676]#Temperature celcus
pH=[780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243]
TGCS=[198, 226755, 227013]#Total Glasgow coma score
CRR=[3348,115,223951,8377,224308]#Peripheral capillary refill rate


# In[ ]:


features=np.array(SpO2+HR+RR+SBP+DBP+FiO2+Glucose+Temp_F+Temp_C+pH+TGCS+CRR)


# In[ ]:


#Select only rows whose ITEMID is in features
chartevents=chartevents.loc[chartevents.ITEMID.isin(features)]

#Select only the necessary columns
chartevents=chartevents[['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','VALUE','VALUENUM']]

chartevents.head(3)


# In[ ]:


#Save the new charevents dataframe
chartevents.to_csv('data/charevents_sorted.csv',index=False)


# In[ ]:


#Load outputevents csv file from  MIMIC-3 database
outputevents=pd.read_csv('data/OUTPUTEVENTS.csv',engine='c',low_memory=False)

#Select only the necessary columns
outputevents=outputevents[['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','VALUE']]

outputevents.head()


# In[ ]:


#Urine Output
UO=[43647,40428 , 41857 , 42001 , 42362 , 42676 , 43171 , 43173 , 42042 , 42068 , 42111 , 42119 , 40715 , 40056 , 40061 , 40085 , 40094 , 40096 , 43897 , 43931 , 43966 , 44080 , 44103 , 44132 , 44237 , 43348 , 43355 , 43365 , 43372 , 43373 , 43374 , 43379 , 43380 , 43431 , 43462 , 43522 , 44706 , 44911 , 44925 , 42810 , 42859 , 43093 , 44325 , 44506 , 43856 , 45304 , 46532 , 46578 , 46658 , 46748 , 40651 , 40055 , 40057 , 40065 , 40069 , 44752 , 44824 , 44837 , 43576 , 43589 , 43633 , 43811 , 43812 , 46177 , 46727 , 46804 , 43987 , 44051 , 44253 , 44278 , 46180 , 45804 , 45841 , 45927 , 42592 , 42666 , 42765 , 42892 , 43053 , 43057 , 42130 , 41922 , 40473 , 43333 , 43347 , 44684 , 44834 , 43638 , 43654 , 43519 , 43537 , 42366 , 45991 , 43583]

#Select Urine Ouput data from  outputevents
outputevents=outputevents.loc[outputevents.ITEMID.isin(UO)]


# In[ ]:


#Save the new outputevents dataframe
outputevents.to_csv('data/outputevents_sorted.csv',index=False)