import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\DELL\Desktop\healthcare.csv")
print(df.head())
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Days hospitalized'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
print(df.head())
df = df[[ 'Age','Billing Amount','Room Number', 'Gender', 'Blood Type', 'Medical Condition','Insurance Provider', 'Admission Type', 'Medication', 'Test Results','Days hospitalized']]
print(df.head())
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
for col in df.columns:
    if col!= ['Age','Billing Amount','Room Number','Days Hospitalized']:
        df[col] = lc.fit_transform(df[col])
print(df.head())
# KNN
x=df.drop(['Room Number','Billing Amount','Days hospitalized','Test Results'],axis=1)
y=df['Test Results']
from sklearn import preprocessing
x=preprocessing.StandardScaler().fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=42)
ks=25
mean_acc=np.zeros((ks-1))
# train as well as predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
for n in range(1,ks-1):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1]=accuracy_score(y_test,yhat)
print(mean_acc)
print("best accuracy was with",mean_acc.max(),"with k=",mean_acc.argmax())
from sklearn.neighbors import KNeighborsClassifier
knnmodel=KNeighborsClassifier(n_neighbors=21)
knnmodel.fit(x_train,y_train)
y_predict1=knnmodel.predict(x_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_predict1)
print(acc)
import pickle
pickle.dump(knnmodel,open("model.pkl","wb"))