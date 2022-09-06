from torch.utils import data
import csv

class Genrator(data.Dataset):
    def __init__(self):
        self.str_list = []
        self.score_list = []
        self.date_list = []
        self.size = 0
        with open('Apple-Twitter-Sentiment.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['sentiment']!='' and row['sentiment']!='not_relevant':
                    s = int(row['sentiment'])
                    if s == 1:
                        s = 0
                    elif s == 3:
                        s = 1
                    elif s == 5:
                        s = 2
                    else:
                        s = 1
                    self.score_list.append(s)
                    self.str_list.append(row['text'])
                    self.date_list.append(row['date'])
                    self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.str_list[index], self.score_list[index],self.date_list[index]

