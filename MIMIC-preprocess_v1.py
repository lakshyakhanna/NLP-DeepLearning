# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 17:59:54 2017

@author: lakshya.khanna
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:01:16 2017

@author: lakshya.khanna
"""


import os
import pandas as pd
import pandasql as pdsql
import numpy as np


pysql = lambda q: pdsql.sqldf(q, globals())

os.chdir("E:\POC\Hospital Readmission POC\MIMIC Data")


#import admission
admission = pd.read_csv('ADMISSIONS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
admission['SEQUENCE'] = admission.sort_values(by =['SUBJECT_ID','ADMITTIME','HADM_ID'],ascending = [True,True,True]).groupby(['SUBJECT_ID']).cumcount() + 1
admission.head()
admission.info()
admission.shape
admission.columns
admission.loc[admission.SUBJECT_ID==36,['ROW_ID','SUBJECT_ID','SEQUENCE']]


#import services
services = pd.read_csv('SERVICES.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
services.info()
services.loc[services.HADM_ID == 191941]


#import ICUstays
icustays = pd.read_csv('ICUSTAYS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
icustays.info()

#checking if there is only one ICU stay per admission
diff = icustays['HADM_ID'].count() - len(icustays['HADM_ID'].unique())
if diff > 0:
    print("There are " + str(diff) + " admissions have more than one ICU stays")
else:
    print("There is only one ICU stay per admission")

temp_icu_stay = icustays[['HADM_ID', 'ICUSTAY_ID', 'LOS']].groupby(by=['HADM_ID'], as_index=False).agg({"ICUSTAY_ID":"max", "LOS":"sum"})
recent_icu_stay = pd.merge(icustays, temp_icu_stay, on=['HADM_ID', 'ICUSTAY_ID'])[['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'LOS_y']]
recent_icu_stay.columns = ['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'LOS']

#Validation
recent_icu_stay.loc[recent_icu_stay.HADM_ID == 147559]
icustays.loc[icustays.HADM_ID == 147559]


#import patients
patients = pd.read_csv('PATIENTS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
patients.info()
patients.columns


#import drgcodes
drgcodes = pd.read_csv('DRGCODES.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
drgcodes.info()
len(drgcodes.HADM_ID.unique())
drgcodes[drgcodes.HADM_ID == 100006]
drgcodes_sev_mor = drgcodes[['HADM_ID','DRG_SEVERITY','DRG_MORTALITY']].groupby('HADM_ID').agg({"DRG_SEVERITY":"mean", "DRG_MORTALITY":"mean"}).reset_index()
drgcodes_sev_mor.loc[drgcodes_sev_mor.DRG_SEVERITY.isnull(),['DRG_SEVERITY']] = 0
drgcodes_sev_mor.loc[drgcodes_sev_mor.DRG_MORTALITY.isnull(),['DRG_MORTALITY']] = 0
drgcodes_sev_mor.shape
drgcodes_sev_mor.columns

"""
#create admission_count
admission_count = admission.groupby('SUBJECT_ID').SUBJECT_ID.agg(['count']).reset_index()
admission_count.columns = ['SUBJECT_ID', 'READMISSION_COUNT']
admission_count.head()
admission_count.info()
admission_count.shape
admission_count[admission_count.READMISSION_COUNT>1].count()# 7537 patients that are readmitted
#admission_count = pysql('SELECT SUBJECT_ID, COUNT(SUBJECT_ID) AS COUNT FROM admission group by SUBJECT_ID;')


#create patients_readmitted
patients_readmitted = pd.merge(admission, admission_count,how = 'left', on = ['SUBJECT_ID'])
patients_readmitted = patients_readmitted.loc[patients_readmitted.READMISSION_COUNT > 1,]
patients_readmitted.shape
patients_readmitted.info()
patients_readmitted.head()
#patients_readmitted = pysql('SELECT a.* , b.COUNT AS READMISSION_COUNT FROM admission a left outer join admission_count b on a.SUBJECT_ID = b.SUBJECT_ID;')
"""

#Get Discharge Time
patients_readmitted_readdays = pysql('select a.SUBJECT_ID, a.HADM_ID,a.DISCHTIME as CURR_DISCHTIME,b.ADMITTIME as NXT_ADMITTIME ,(julianday(b.ADMITTIME) - julianday(a.DISCHTIME)) as DAYS_TO_READMISSION  from admission a inner join admission b on a.SUBJECT_ID = b.SUBJECT_ID  and a.SEQUENCE = b.SEQUENCE - 1')
patients_readmitted_readdays.shape
patients_readmitted_readdays = patients_readmitted_readdays.loc[patients_readmitted_readdays.DAYS_TO_READMISSION>0.0]



#patients admiited which have taken cardiac services
patients_readmitted_heart = pysql('SELECT distinct a.HADM_ID from patients_readmitted_readdays a WHERE a.HADM_ID in (SELECT DISTINCT HADM_ID FROM services where CURR_SERVICE in (\'CMED\',\'CSURG\'));')
patients_readmitted_heart.shape
patients_readmitted_heart.info()
len(patients_readmitted_heart.HADM_ID.unique())
hrt_adm_id = list(patients_readmitted_heart.HADM_ID.unique())

patients_readmitted_readdays['CHF_FLAG'] = patients_readmitted_readdays['HADM_ID'].apply(lambda x : 1 if x in hrt_adm_id else 0)
patients_readmitted_readdays[['HADM_ID','CHF_FLAG']]
hrt_adm_id.__contains__([162391])



"""
#no of surgeries
surgeries = pysql('SELECT count(1) as NO_OF_SURGERIES, CURR_SERVICE,HADM_ID FROM services where CURR_SERVICE in (\'CSURG\') GROUP BY CURR_SERVICE,HADM_ID  ;')
surgeries.shape
surgeries[surgeries['NO_OF_SURGERIES']>1]
test = surgeries.groupby('HADM_ID').count()
test.shape

#Add no of surgeries to the dataset

patients_readmitted_heart_surgeries = pd.merge(patients_readmitted_heart,surgeries[['HADM_ID','NO_OF_SURGERIES']],how = 'left' , on = ['HADM_ID'])
patients_readmitted_heart_surgeries.NO_OF_SURGERIES.unique()
patients_readmitted_heart_surgeries.NO_OF_SURGERIES.fillna(0.0,inplace = True)
patients_readmitted_heart_surgeries['NO_OF_SURGERIES'] = patients_readmitted_heart_surgeries.NO_OF_SURGERIES.astype('int64')
patients_readmitted_heart_surgeries.info()
patients_readmitted_heart_surgeries.shape"""


#Merging the dataframe with admission to get the admission attributes
#patients_readmit_admattr = pd.merge(patients_readmitted_heart,admission[['HADM_ID','ADMISSION_TYPE','ADMITTIME', 'ADMISSION_LOCATION','DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION','MARITAL_STATUS', 'ETHNICITY']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admattr = pd.merge(patients_readmitted_readdays,admission[['HADM_ID','ADMISSION_TYPE','ADMITTIME', 'ADMISSION_LOCATION','DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION','MARITAL_STATUS', 'ETHNICITY']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admattr.shape 
total_missing = patients_readmit_admattr.isnull().sum()
patients_readmit_admattr.columns


#Merging the dataframe with patient to get the patient attributes
patients_readmit_admPat_attr = pd.merge(patients_readmit_admattr,patients[['SUBJECT_ID','GENDER', 'DOB']] ,how = 'left' , on = ['SUBJECT_ID'])
patients_readmit_admPat_attr.shape
patients_readmit_admPat_attr.columns
patients_readmit_admPat_attr.isnull().sum()


#Merging the dataframe with drgcodes to get the mortality & severity scores attributes
patients_readmit_admPatDrg_attr = pd.merge(patients_readmit_admPat_attr,drgcodes_sev_mor[['HADM_ID', 'DRG_SEVERITY', 'DRG_MORTALITY']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admPatDrg_attr.shape
patients_readmit_admPatDrg_attr.columns
patients_readmit_admPatDrg_attr.isnull().sum()

tmp = pysql('SELECT HADM_ID from patients_readmit_admPat_attr where HADM_ID not in (select HADM_ID from drgcodes);')


#Merging the dataframe with ICUstay to get the LOS,Last_icu scores attributes
patients_readmit_admPatDrgICU_attr = pd.merge(patients_readmit_admPatDrg_attr,recent_icu_stay[['HADM_ID','LAST_CAREUNIT', 'LOS']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admPatDrgICU_attr.shape
patients_readmit_admPatDrgICU_attr.columns

patients_readmit_admPatDrgICU_attr[patients_readmit_admPatDrgICU_attr.DAYS_TO_READMISSION >= 30].shape
patients_readmit_admPatDrgICU_attr['READMISSION_60dAYS'] = patients_readmit_admPatDrgICU_attr.DAYS_TO_READMISSION.apply(lambda x : 1 if x <= 60 else 0)
patients_readmit_admPatDrgICU_attr['READMISSION_30dAYS'] = patients_readmit_admPatDrgICU_attr.DAYS_TO_READMISSION.apply(lambda x : 1 if x <= 30 else 0)

final_dataset1  = patients_readmit_admPatDrgICU_attr.drop(['CURR_DISCHTIME','NXT_ADMITTIME','ADMITTIME','DOB','READMISSION_30dAYS'],axis = 1)

# Impute Values for missing data
final_dataset1.loc[final_dataset1.RELIGION.isnull(),['RELIGION']] = final_dataset1.RELIGION.mode()[0]
final_dataset1.loc[final_dataset1.MARITAL_STATUS.isnull(),['MARITAL_STATUS']] = final_dataset1.MARITAL_STATUS.mode()[0]
final_dataset1.loc[final_dataset1.LAST_CAREUNIT.isnull(),['LAST_CAREUNIT']] = final_dataset1.LAST_CAREUNIT.mode()[0]
final_dataset1.loc[final_dataset1.LOS.isnull(),['LOS']] = 0
final_dataset1.loc[final_dataset1.DRG_SEVERITY.isnull(),['DRG_SEVERITY']] = 0
final_dataset1.loc[final_dataset1.DRG_MORTALITY.isnull(),['DRG_MORTALITY']] = 0

final_dataset1.columns
final_dataset1.isnull().sum()

final_dataset1.sort_values(['SUBJECT_ID','HADM_ID'], inplace = True)
final_dataset1.head(30)

final_dataset1.head().values.tolist()
final_dataset1.dtypes
final_dataset1.columns
tmp1 = final_dataset1.head(150).values.tolist()

final_dataset1.to_csv('Keras_dataset.csv',index=False)

final_dummy_dataset = pd.get_dummies(final_dataset1,columns=['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION','INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'GENDER', 'LAST_CAREUNIT'])

def make_patient_adm_list(dataset):
    pat = []
    adm = []
    pat_ctr = 0
    patient = -1
    for i in dataset.head(100).values.tolist():
        if patient != i[0]:
            adm.append(i)
            adm_final = list(adm)
            pat.insert(pat_ctr,adm_final)
            pat_ctr += 1
            patient = i[0]
            adm.clear()
        else:
            adm_final.append(i)
    return pat


   
def patient_adm_factorial(pat_adm_list):
    new_adm = []
    new_pat = []
    new_pat_ctr = 0
    for i in pat_adm_list:
        for j,k in enumerate(i):
            if k[3] == 1:
                new_pat.insert(new_pat_ctr, i[0:j+1])
                new_pat_ctr += 1
                new_adm.clear()
    return new_pat


def get_y_array(pat_adm_combination):
    y_train = []
    sub_id = []
    y_train_ctr = 0
    for i in pat_adm_combination:
        y_train.insert(y_train_ctr,i[-1][-1])
        sub_id.insert(y_train_ctr,i[-1][0])    
        y_train_ctr += 1
    return y_train

   


def get_max_seq_length(pat_adm_combination):
    max_seq_length = 0
    for i in pat_adm_combination:
        if len(i) > max_seq_length:
            max_seq_length = len(i)
    return max_seq_length
    
pat = make_patient_adm_list(final_dummy_dataset)
pat_data_set = patient_adm_factorial(pat) 
y_train = get_y_array(pat_data_set)
print(pat_data_set)
print (y_train)   
max_len = get_max_seq_length(pat_data_set)


for i in pat_data_set:
    for j in i:
        #del j[0:2]
        #del j[-1]
        print(len(j))
print(pat_data_set)

import numpy as np    
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,KFold

np.random.seed(25)

print(pat_data_set)
X_train = list(pat_data_set)
y_train = get_y_array(pat_data_set)
y_train_dummy = to_categorical(y_train)

max_seq_len = get_max_seq_length(pat_data_set)
data =  pad_sequences(pat_data_set,max_seq_len)
#create the model
def base_model():
    model = Sequential()   
    model.add(LSTM(100,input_dim =3))
    model.add(Dense(32,input_dim = 370,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#history = model.fit(X_train, y_train_dummy, verbose=0, epochs=20)
estimator = KerasClassifier(build_fn=base_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=12)


results = cross_val_score(estimator, X_train, y_train_dummy, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimator.fit(np.array(X_train),np.array(y_train_dummy))

y_pred = estimator.predict(np.array(X_test))

score = accuracy_score(y_pred,y_test)



    
         


                

a = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12],[4,5,4,54],[5]]]
a[0][1][2]
for i in a:
    for j in i:
        #del j[0:2]
        del j[-1]
print (a)

b = np.array(a)
b
be = [1,2,3,4]
be[0:0]
be[-1]
be.reverse()
print(be)

max_seq_length = 0
for i in a:
    if len(i) > max_seq_length:
        max_seq_length = len(i)
print(max_seq_length)
