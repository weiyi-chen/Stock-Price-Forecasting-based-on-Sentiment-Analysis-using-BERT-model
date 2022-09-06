from transformers import AutoTokenizer, AutoModel
from vanilla_classifier import Classifier3
from data_generator import Genrator
import torch.nn as nn
import torch
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
params = {"batch_size": 4000, "shuffle": True, "num_workers": 1}
gen = Genrator()
print("total data size :",gen.size)
training_generator = data.DataLoader(gen, **params)

classifier = Classifier3().to(device)
crossentropy_loss = nn.CrossEntropyLoss() 
learning_rate = 3e-4
model_file_name = 'classifier3_relu.pth'
exist_model_name = '39_classifier3_relu.pth'
optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-5)
classifier.load_state_dict(torch.load("./models/" + exist_model_name))
for epoch in range (40):
    idx = 0
    for x,y in training_generator:
        idx += 1
        with torch.no_grad():
            inputs = tokenizer(list(x), return_tensors="pt",padding=True, truncation=True) #[batch 100, longest sent 54, feature 768]
            outputs = model(**inputs) # [batch 100, longest sent 54, feature 768]
            output_s = outputs[0].permute(1,0,2)
        y_hat = classifier(output_s[0].to(device))
        y = torch.FloatTensor(list(y)).to(device)
        # y -= 1
        # y = y
        # loss_1 = crossentropy_loss(y_hat,torch.round(y).to(torch.long))
        out = crossentropy_loss(y_hat,torch.round(y).to(torch.long))
        # y_l = torch.argmax(y_hat, dim=1)
        # y = torch.unsqueeze(y, dim=1)
        # out = mse_loss(y_l.to(device), y)+ loss_1
        optimizer.zero_grad()
        out.backward()
        optimizer.step()
        print("train loss:{:.10f} in {} iteration {} epoch.".format(out.data, idx , epoch))
    torch.save(classifier.state_dict(), "./models/"+str(epoch)+"_" + model_file_name)