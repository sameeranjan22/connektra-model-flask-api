from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from predict import evaluate_one_text
import sys
import json
import pandas as pd
import torch
import numpy as np
import requests, json
from transformers import BertTokenizerFast, BertForTokenClassification
import tempfile

class BertModel(torch.nn.Module):

    def __init__(self):
        unique_labels = ['I-trigger', 'B-source', 'I-action', 'B-destination', 'B-trigger', 'B-action', 'O']
        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output

app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument(name='sentence',
                    type=str,
                    required=True,
                    help='Provide a sentence')

class Predict(Resource):
    def post(self):
        unique_labels = ['I-trigger', 'B-source', 'I-action', 'B-destination', 'B-trigger', 'B-action', 'O']
        model = BertModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = 'model'
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
        args = parser.parse_args()
        prompt = args["sentence"]
        #print(prompt)
        #print(sentence)
        results = evaluate_one_text(model, prompt)
        #print(results)
        output = {'classification': [], }
        #print(results)
        for variable, name in results.items():
            output['classification'].append({variable: name})
        #print(output)
        json_output = json.dumps(output)
        return json_output

#curl -d '{"sentence": "Create a link between my new mention located in slack and get user in jira."}' -H 'Content-Type: application/json' localhost:5000/predict
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    
    app.run(debug=True)