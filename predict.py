import sys
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification

class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


def align_word_ids(texts):
    unique_labels = ['I-trigger', 'B-source', 'I-action', 'B-destination', 'B-trigger', 'B-action', 'O']
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def evaluate_one_text(model, sentence):
    unique_labels = ['I-trigger', 'B-source', 'I-action', 'B-destination', 'B-trigger', 'B-action', 'O']
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    result = []
    result.append(sentence)
    result.append(prediction_label)
    source = ""
    destination = ""
    trigger = ""
    action = ""
    sentence_list = sentence.split(" ")
    for i in range(len(sentence_list)):
        if(prediction_label[i]=="B-source"):
            source+=sentence_list[i]
        if(prediction_label[i]=="B-destination"):
            destination+=sentence_list[i]
        if(prediction_label[i]=="B-trigger"):
            trigger+=sentence_list[i]
        if(prediction_label[i]=="B-action"):
            action+=sentence_list[i]
        if(prediction_label[i]=="I-source"):
            source+=" "
            source+=sentence_list[i]
        if(prediction_label[i]=="I-destination"):
            destination+=" "
            destination+=sentence_list[i]
        if(prediction_label[i]=="I-trigger"):
            trigger+=" "
            trigger+=sentence_list[i]
        if(prediction_label[i]=="I-action"):
            action+=" "
            action+=sentence_list[i]
    return {"source":source, "destination":destination, "trigger":trigger, "action":action}

if __name__ == '__main__':
    unique_labels = ['I-trigger', 'B-source', 'I-action', 'B-destination', 'B-trigger', 'B-action', 'O']
    model = BertModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'model'
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    sentence = "Build a connection between contact trigger originating from bitrix24 and list all customers taking place in zoho."
    [sentence, label] = evaluate_one_text(model, sentence)
    sentence_list = sentence.split(" ")
    for i in range(len(sentence_list)):
        print(sentence_list[i], label[i])