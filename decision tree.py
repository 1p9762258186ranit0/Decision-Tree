# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:31:57 2023

@author: lenovo
"""



           1]PROBLEM:  'Company_Data.csv'


    
BUSINESS OBJECTIVE:-    
Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  




#Importing the Necessary Liabrary
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Decision Tree/Company_Data.csv')

#EDA
df.info()
df.head()
df.tail()
df.describe()
df.shape

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['ShelveLoc']=l.fit_transform(df['ShelveLoc'])
df['Urban']=l.fit_transform(df['Urban'])
df['US']=l.fit_transform(df['US'])

#Discritization for target variable(Sales)
df['Salaries_nefw'] = pd.cut(df['Sales'], bins=[min(df.Sales) - 1, 
                                                  df.Sales.mean(), max(df.Sales)], labels=["Low","High"])


df.head()
df.drop(['Sales'],axis=1,inplace=True)

inpu=df.iloc[:,0:10]#Predictors
target=df.iloc[:,[10]]#Target

#Split the Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.2)

#Model Building
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion='entropy')
model.fit(x_train,y_train)

#Evaluations On Train data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
train_report=classification_report(y_train,trainpred)
confusion_matrix(y_train,trainpred)

#Evaluations On Test data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix(y_test,testpred)



           2]PROBLEM ::'Fraud_check.csv'

BUSINESS OBJECTIVE:-Use decision trees to prepare a model on fraud data. 

Note=(Treating those who have taxable_income <= 30000 as "Risky" and others are "Good")






#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Decision Tree/Fraud_check.csv')

#EDA
df.describe()
df.head()
df.tail()
df.shape
df.info()
df.isna().sum()


df=df.rename(columns={'Taxable.Income':'tax'})#Rename 
df.info()


#Creating Discritization..>=3000 as 'Risky' and remainings are good.
ma=['Risky','Good']
bi=[0,30000,99619]
df['tax_new']=pd.cut(x=df.tax,bins=bi,labels=ma,retbins=False,duplicates='raise',ordered=True)
df.head()

df.drop(['tax'],axis=1,inplace=True)#Dropping the original 'tax' feature,which is continuous in nature.

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['Undergrad']=l.fit_transform(df['Undergrad'])
df['Marital.Status']=l.fit_transform(df['Marital.Status'])
df['Urban']=l.fit_transform(df['Urban'])


inpu=df.iloc[:,0:5]#Predictors
target=df.iloc[:,[5]]#Target

#Split the Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.2)

#Model Building
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion='entropy',min_samples_split=30,min_samples_leaf=18)
#I tune/Prunning the model again and again and i found ,min_samples_split=45,min_samples_leaf=28 is the optimum number for achieving the good accuracy.
model.fit(x_train,y_train)


#Evalutions on Train Data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
train_report=classification_report(y_train,trainpred)
confusion_matrix(y_train,trainpred)


#Evalutions on Train Data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix(y_test,testpred)
