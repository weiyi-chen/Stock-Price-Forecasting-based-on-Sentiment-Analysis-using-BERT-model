import datetime
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
##get time
os.chdir('/Users/chenweiyi/Desktop/last/coding')
datestart = '2021-06-02'
dateend = '2022-06-02'
datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
date_list = []
date_list.append(datestart.strftime('%Y-%m-%d'))
while datestart<dateend:
    datestart+=datetime.timedelta(days=+1)
    date_list.append(datestart.strftime('%Y-%m-%d'))
print(len(date_list))

##get stock trend list
close_list = []
open_list = []

date_stock_list =[]
with open('AAPL_stock.csv',newline = '') as file:
    df = csv.DictReader(file)
    for row in df:
        if row['Date']<"2022-06-03" and row['Date']>"2021-06-01":
            date_stock_list.append((row['Date']))
            open_list.append(float(row['Open']))
            close_list.append(float(row['Close']))

trend_list = []
for i in range(len(close_list)-1):
    trend_list.append(close_list[i+1]-close_list[i])

input_list = copy.deepcopy(trend_list)

with open("final_aapl.txt", "r") as f:  
    lines = [float(l[25:31]) for l in f]

filtered_score_list = []
for d,l in zip(date_list,lines):
    if d in date_stock_list:
        filtered_score_list.append(l)

input_list.pop(0)
input_list.pop(-1)
del trend_list[0:2]

## normalization
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

stock_trend = normalization(trend_list)
filtered_score_list = normalization(filtered_score_list)

## set x and y
target =[]
for i in range(len(trend_list)):
    if trend_list[i] >0:
        target.append(1)
    if trend_list[i]<0:
        target.append(0)
    if trend_list == 0:
        target.append(-1)

x = []
for d,c in zip(input_list, filtered_score_list):
    x.append([d,c])
x = np.array(x)
## nn model building
X_train, X_test, y_train, y_test = train_test_split(x,target, test_size=0.2,random_state = 0)
dnn = MLPClassifier(hidden_layer_sizes=(2),random_state = 420) # two layers
dnn = dnn.fit(X_train,y_train)
pred_dnn = dnn.predict(X_test)
dnn_val=cross_val_score(dnn,x,target,cv=5)
cm = confusion_matrix(y_test, dnn.predict(X_test))
print(cm)
