import pandas as pd
import pickle 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
df=pd.read_csv('dataset\Crop_recommendation.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
try:
    pickle.dump(model, open("model.pkl", "wb"))
    print("Model loaded")
except:
    print("There is some issue")

#y_pred=model.predict(x_test)
#print(accuracy_score(y_test,y_pred))
#print(classification_report(y_test,y_pred))