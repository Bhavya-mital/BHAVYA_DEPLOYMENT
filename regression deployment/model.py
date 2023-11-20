import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\DELL\Desktop\winequalityred.csv")
print(df.head())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
X = df.drop(['quality'],axis=1)
y = df['quality']
# RANDOM FOREST
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state= 42)
rfc = RandomForestRegressor()
rfc.fit(X_train,y_train)
y_rfc_pred = rfc.predict(X_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_rfc_pred)
print(r2)
print(X.shape)
print(y.shape)
import pickle
pickle.dump(rfc, open("model.pkl","wb"))
