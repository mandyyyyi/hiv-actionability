import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertModel
from transformers import AdamW
from tqdm import tqdm, trange
import json 
import numpy as np

def bert_run(train_file, train_content, train_label, file_name, output, content, id, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_texts=[]
    train_labels=[]

    dev_texts=[]
    dev_labels=[]
    data=pd.read_csv(file_name)
    result_id=[]
    for ele in data.to_dict('records'):
        if not isinstance(ele[content], str):
            continue
        dev_texts.append(ele[content])
        dev_labels.append(0)
        result_id.append(int(eval(ele[id])))
    # file_name="/home/xw27/dataset/hiv_tweet/hiv-spanishactionabilityclassifier/data/full_dataset_with_favourites.csv"
    data=pd.read_csv(train_file)
    for ele in data.to_dict('records'):
        if int(ele[train_label]) == -1:
            train_labels.append(0)
            train_texts.append(str(ele[train_content]))
        else:
            train_labels.append(int(ele[train_label]))
            train_texts.append(str(ele[train_content]))

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def load_data(train_texts, train_labels, dev_texts, dev_labels):
        train_encodings = tokenizer(train_texts,truncation=True,max_length=50, padding=True)
        dev_encodings = tokenizer(dev_texts, truncation=True,max_length=50, padding=True)
        train_dataset = Dataset(train_encodings, train_labels)
        dev_dataset = Dataset(dev_encodings, dev_labels)
        return train_dataset, dev_dataset

    train_dataset, dev_dataset = load_data(train_texts, train_labels, dev_texts, dev_labels)

    train_loader= DataLoader(train_dataset, shuffle=True, batch_size=8)
    dev_loader= DataLoader(dev_dataset, shuffle=False, batch_size=8)
    
    optim = AdamW(model.parameters(), lr=1e-5,eps=1e-4)

    model.to(device)
    model.train()
    for epoch in trange(1, desc="Epoch"):
        avg_loss = []
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration", miniters=int(2943/100))):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels = labels, return_dict=True)
            loss = outputs.loss
            avg_loss.append(loss.item())
            loss.backward()
            optim.step()
        
    print("Epoch %d loss =" %epoch, np.mean(avg_loss))
    model.to(device)
    model.eval()

    predictions=[]
    true_labels=[]
    score=[]
    for step, batch in enumerate(tqdm(dev_loader, desc="Iteration", miniters=int(2943/100))):
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,return_dict=True)
            
        logits = outputs.logits 

            # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        score.extend([row[1] for row in logits])
        logits=np.argmax(logits, axis=1).flatten()
        
        label_ids = list(labels.to('cpu').numpy())
        predictions.extend(logits)
        true_labels.extend(label_ids)
    #start

    df = pd.DataFrame({'tweet_ids': result_id, 'prediction':predictions,
                    'content': dev_texts, 'model': ["bert"]*len(dev_texts)})
    df.to_csv(output)
    
    #end




