from transformers import AutoTokenizer, AutoModel
from vanilla_classifier import Classifier3
from data_cleaner import TestDataGenerator
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

classifier = Classifier3().to(device)
g = TestDataGenerator()
model_file_name = '1399_classifier3_relu.pth'
classifier.load_state_dict(torch.load("./models/" + model_file_name))

xx_list = []
date_dict = dict()
last_d = 0
x_list =[]
date_list = []
for d,t in g:
    if last_d != d:
        if len(x_list)!= 0:
            t_list = copy.deepcopy(x_list)
            xx_list.append(t_list)
            date_list.append(d)
        x_list.clear()
        x_list.append(t)
        last_d = d
    else:
        x_list.append(t)

print("date_list : ",len(date_list))
print("xx_list : ",len(xx_list))

score_list = []
for x_list in xx_list:
    avg_score = 0
    n = 0
    for x in x_list:
        with torch.no_grad():
            inputs = tokenizer(x, return_tensors="pt")
            outputs = model(**inputs) # [batch 100, longest sent 54, feature 768]
            y_hat = classifier(outputs[0][0][0].to(device))
            print(y_hat)
            y_l = torch.argmax(y_hat, dim=0)
            print("twitter :",x)
            print("pred :",y_l)
            avg_score += (y_l-1)
            n += 1
    if n!=0:
        avg_score = (avg_score)/float(n)
        print("#########  avg score : ",avg_score)
        score_list.append(avg_score)

print("Final scores : ")
for d,s in zip(date_list,score_list):
    print(d,"  ",s)

