# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program. 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by:GOWRISANKAR P
RegisterNumber:212222230041
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
## Encoding:
![Screenshot 2024-11-14 114506](https://github.com/user-attachments/assets/4d9c3881-d84d-468f-95e8-eb2c63b6882d)
## Head():
![Screenshot 2024-11-14 114713](https://github.com/user-attachments/assets/2f51997d-bcc3-4d02-80e3-3de0ed89caf2)
## Info():
![Screenshot 2024-11-14 114831](https://github.com/user-attachments/assets/c8edb60c-0d86-44df-a11f-568c523b45b2)
## isnull().sum():
![Screenshot 2024-11-14 114944](https://github.com/user-attachments/assets/f2e986e7-ec95-4458-8943-44f6adca97bd)

## Prediction of y:
![Screenshot 2024-11-14 115047](https://github.com/user-attachments/assets/58a3c16f-5be3-4795-974c-cdc245b85f65)
## Accuracy:
![Screenshot 2024-11-14 115140](https://github.com/user-attachments/assets/4df46bb8-178e-44d1-b537-8cb664f5b851)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
