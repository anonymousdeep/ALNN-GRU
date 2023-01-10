#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time as tm
import math


# In[ ]:


#Define the path where the file is located
path='../../../../../data/mimic3/'


# In[ ]:


#Load charevents_{prior_hours}.cvs file
prior_hours=48
chartevents=pd.read_csv(f'../data/chartevents_{prior_hours}.csv',low_memory=False,engine='c',parse_dates=['CHARTTIME'])
chartevents.head(2)


# In[ ]:


#Get all admissions id HADM_ID
hadm_ids=chartevents.HADM_ID.unique()


# In[ ]:


# pd.DataFrame(data=hadm_ids).to_csv('hadmid.csv',index=False)


# In[ ]:


# Imputation
#We gradually save the sub dataframe to speed up the iputation process
datafram_tempo=pd.DataFrame()
compteur=1
start= tm.perf_counter()
for idx,k in enumerate(hadm_ids):
    sub_dataframe=chartevents.loc[chartevents['HADM_ID']==k,:].sort_values(by="CHARTTIME")
    list_itemid=sub_dataframe.ITEMID.unique()
    for itm in list_itemid:
        if (itm!=3348):
            sub_sub_dataframe=sub_dataframe.loc[sub_dataframe['ITEMID']==itm,:]
            sub_sub_dataframe=sub_sub_dataframe.assign(MASK=sub_sub_dataframe['VALUENUM'].apply(lambda x: 0 if math.isnan(x) else 1))
            sub_sub_sub_dataframe=sub_sub_dataframe[['CHARTTIME','VALUENUM']]
            sub_sub_sub_dataframe.set_index('CHARTTIME',inplace=True)
            sub_sub_sub_dataframe=sub_sub_sub_dataframe.interpolate(method="time")
            sub_sub_sub_dataframe=sub_sub_sub_dataframe.fillna(method="bfill")
            sub_sub_dataframe=sub_sub_dataframe.assign(VALUENUM=sub_sub_sub_dataframe['VALUENUM'].values)
        else:
            sub_sub_dataframe=sub_dataframe.loc[sub_dataframe['ITEMID']==itm,:] 
            sub_sub_dataframe=sub_sub_dataframe.assign(MASK=sub_sub_dataframe['VALUENUM'].apply(lambda x: 0 if math.isnan(x) else 1))

         
        
        datafram_tempo=pd.concat([datafram_tempo,sub_sub_dataframe],axis=0)
        
    #We save by block to speed up the process 
    if (idx%3000==0) or (idx==(len(hadm_ids)-1)):
        #Monitoring steps
        print(compteur,'-',idx)
        datafram_tempo.to_csv(f'{path}chartevents_{prior_hours}_mixt_interpolate_bfill_{compteur}.csv',index=False)
        compteur=compteur+1;
        datafram_tempo=pd.DataFrame()
        
finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")


# In[ ]:


#Concatenate all blocks
datafram_tempo=pd.DataFrame()
for k in range(compteur-1):
    sub_data_frame=pd.read_csv(f'{path}chartevents_{prior_hours}_mixt_interpolate_bfill_{k+1}.csv',low_memory=False,engine='c')
    datafram_tempo=pd.concat([datafram_tempo,sub_data_frame],axis=0)

#Save the new chartevents dataframe imputed
datafram_tempo.to_csv(f'{path}chartevents_{prior_hours}_imputed.csv',index=False)
datafram_tempo.head()


# In[ ]:


#To save memory
# del datafram_tempo

