import csv
from torch.utils import data


class TestDataGenerator(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.total_data_list = []
        unique_dict = dict()
        date_dict = dict()
        file_list = ["./nvda1.csv"]
        # with open('aapl4.csv', newline='') as csvfile:
        for filename in file_list:
            with open(filename, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # print(row['date'], row['tweet'])    
                    if row['id'] not in unique_dict.keys():
                        unique_dict[row['id']] = 1
                        if row['language'] == 'en':
                            if row['date'] not in date_dict.keys():
                                date_dict[row['date']] = 1
                            item = [self.date_string_to_int(row['date']),row['tweet']]
                            self.total_data_list.append(item)

        self.total_data_list.sort(key=self.date_func)

    def date_string_to_int(self,str):
        s =str[0:4]+str[5:7]+str[8:]# 2021-06-01
        return int(s)

    def date_func(self, e):
        return e[0]

    def __len__(self):
        return len(self.total_data_list)

    def __getitem__(self,index):
        return self.total_data_list[index][0],self.total_data_list[index][1] # date, tweet

if __name__ == "__main__":
    t = TestDataGenerator()
    print(len(t))