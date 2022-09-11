from pickletools import TAKEN_FROM_ARGUMENT1, TAKEN_FROM_ARGUMENT8U
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
os.chdir('/Users/chenweiyi/Desktop/last/coding')
close_list = []
open_list = []

date_stock_list =[]
with open('TSLA_stock.csv',newline = '') as file:
    df = csv.DictReader(file)
    for row in df:
        if row['Date']<"2022-06-02" and row['Date']>"2021-05-31":
            date_stock_list.append((row['Date']))
            open_list.append(float(row['Open']))
            close_list.append(float(row['Close']))

print("close_list : ",len(close_list))
trend_list = []
for i in range(len(close_list)):
    trend_list.append(close_list[i]-open_list[i])

input_list = copy.deepcopy(trend_list)

with open("final_tesla.txt", "r") as f:  
    lines = [float(l[7:14]) for l in f]

filtered_score_list = []
for d,l in zip(date_stock_list,lines):
    if d in date_stock_list:
        filtered_score_list.append(l)
print('filtered_score_list: ',len(filtered_score_list))



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

stock_trend = normalization(trend_list)
filtered_score_list = normalization(filtered_score_list)

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
#print(x)
"""
build svm model
"""
x = np.array(x)
X = copy.deepcopy(x)
X_train, X_test, y_train, y_test = train_test_split(x,target, test_size=0.2,random_state = 3)
"""
Optimize parameters
"""
result =[]
for i in  np.linspace(0.1,10,30):
    C = 2 ** i
    for j in np.linspace(0.0001,0.5,10):
        G = 2 ** j

    rbf_svc = svm.SVC(kernel='rbf', gamma=G, C=C).fit(X_train, y_train) # 高斯kernel
        # rbf_svc = svm.LinearSVC(C=C).fit(data_np, target_np)
    scores = cross_val_score(rbf_svc, X_test, y_test, cv=10)
    result.append([C,G,scores.mean()])
    result1 = sorted(result, key=lambda x:x[2])

for i in result1:
    print(i)
"""
prediction
"""
precition = rbf_svc.score(X_test,y_test)
print('precision is : ',precition*100,"%")

cm = confusion_matrix(y_test, rbf_svc.predict(X_test))
print(cm)
print(classification_report(y_test, clf.predict(X_test), labels=[1, 2, 3]))