
import datetime
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import pandas as pd

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

"""
Read the data
"""
date_dict = dict()
stock_dict = dict()
analy_tsla = False
if analy_tsla:
    stock_file = "./csv/TSLA_stock_2017.csv"
    score_file = "final_tsla2.csv"
else:
    stock_file = "./csv/AAPL_stock_2017.csv"
    score_file = "final_aapl.csv"

with open(stock_file,newline = '') as file:
    df = csv.DictReader(file)
    for row in df:
        dat = datetime.datetime.strptime(row['Date'] ,'%Y-%m-%d')
        dat = datetime.datetime.strftime(dat ,'%Y%m%d')
        date_dict[dat] = 1
        stock_dict[dat] = float(row['Close'])
        # stock_dict[dat] = float(row['Close'])-float(row["Open"])


score_dict = dict()
with open(score_file,newline = '') as f:  
    df = csv.DictReader(f)
    for row in df:
        dat =datetime.datetime.strptime(row['date'] ,'%Y%m%d')
        dat = datetime.datetime.strftime(dat ,'%Y%m%d')
        if dat not in date_dict.keys():
            date_dict[dat] = 1
        else: 
            date_dict[dat] = 2
            score_dict[dat] = row['score']

filtered_score_list = []
filtered_stock_list = []
date_stock_list = []

for k,v in date_dict.items():
    # print(k,v)
    if v == 2:
        filtered_score_list.append(float(score_dict[k]))
        filtered_stock_list.append(float(stock_dict[k]))
        date_stock_list.append(pd.to_datetime(k))

print("filtered_score_list : ",len(filtered_score_list))
print("filtered_stock_list : ",len(filtered_stock_list))
print("date_stock_list : ",len(date_stock_list))

"""
Normalize the data
"""
start = 0
length = 240
shift = 0
# filtered_score_diff_arr = np.diff(np.array(filtered_score_list)) 
# filtered_score_diff_arr = normalization(filtered_score_diff_arr)
filtered_score_list = np.array(filtered_score_list[start-shift:start+length-shift])
# filtered_score_list = np.array(filtered_score_list[start:])
filtered_score_list = normalization(filtered_score_list)

# filtered_stock_list = np.diff(np.array(filtered_stock_list))
# filtered_stock_list = np.array(filtered_stock_list[start-shift:-shift])
filtered_stock_list = np.array(filtered_stock_list[start:start+length])
filtered_stock_list = normalization(np.array(filtered_stock_list))
corr2 = np.corrcoef(filtered_stock_list,filtered_score_list)

print(corr2)
"""
Plot the result
"""
# date_stock_list = (date_stock_list)[1:]
plt.plot(np.array(date_stock_list[start:start+length]), filtered_stock_list,  c='y', alpha=0.5)
plt.plot(np.array(date_stock_list[start:start+length]), filtered_score_list, c='r', alpha=0.5)
plt.xticks(date_stock_list[start:start+length], rotation=45)
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y%m%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.legend(['stock_trend','predict_trend'])
if analy_tsla:
    plt.title('TSLA')
else:
    plt.title('AAPL')
plt.show()
