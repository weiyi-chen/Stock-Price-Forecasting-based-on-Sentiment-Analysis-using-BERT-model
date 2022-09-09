from data_cleaner import TestDataGenerator
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

g = TestDataGenerator()
tweets = dict()
dates = []
for d,t in g:
    if d not in tweets.keys():
        dates.append(d)
        tweets[d] = 1
    else:
        tweets[d] += 1
# print(tweets)
# dates = ['01/02/1991','01/03/1991','01/04/1991']
x = [dt.datetime.strptime(str(d),'%Y%m%d').date() for d in dates]
# y = range(len(x)) # many thanks to Kyss Tao for setting me straight here
y = [tweets[d] for d in dates]

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.plot(x,y)
plt.gcf().autofmt_xdate()
# plt.legend(['daily tweets of AAPL crawled by twint'])
plt.title('Tweets of AAPL')
plt.show()