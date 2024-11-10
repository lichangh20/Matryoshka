import os
import json
import argparse
import re
# from data.datasets import GeneralSeq2SeqDataset, get_all_labels
from classification import create_metric_f1_accuracy_chatgpt, create_metric_mae_rmse_chatgpt
from generation import create_metric_bleu_rouge_meteor_chatgpt


task = 'LaMP-1'
index = task.split('-')[1].lower()
file_path = f"data/lamp/inference/lamp{index}/inference_dev_data.jsonl"
with open(file_path, 'r') as f:
    data = json.load(f)
name_list = ["gemini_response", "gpt_response"]
name = "gpt_response"

def get_all_labels(task):
    if task == "LaMP-1":
        return ["[1]","[2]"]
    elif task == "LaMP-2N":
        return ['women', 'religion', 'politics', 'style & beauty', 'entertainment', 'culture & arts', 'sports', 'science & technology', 'travel', 'business', 'crime', 'education', 'healthy living', 'parents', 'food & drink']
    elif task == "LaMP-2M":
        return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
    elif task == "LaMP-3":
        return ["1", "2", "3", "4", "5"]
    elif task == "LaMP-4":
        return []
    elif task == "LaMP-5":
        return []
    elif task == "LaMP-6":
        return []
    elif task == "LaMP-7":
        return []


labels = get_all_labels(task)
if task in ['LaMP-1', 'LaMP-2N', 'LaMP-2M']:
    compute_metrics = create_metric_f1_accuracy_chatgpt(all_labels=labels)
elif task == 'LaMP-3':
    compute_metrics = create_metric_mae_rmse_chatgpt(all_labels=labels)
elif task in ['LaMP-4', 'LaMP-5', 'LaMP-6', 'LaMP-7']:
    compute_metrics = create_metric_bleu_rouge_meteor_chatgpt()


preds = []
answers = []


for point in data:
    if not point[name]:
        continue
    if name == "gemini_response":
        point[name] = point[name].replace(' \n', '')
    if task == "LaMP-1":
        preds.append(point[name])
        answers.append(point['answer'])
    elif task == "LaMP-2N" or task == "LaMP-2M":
        for label in labels:
            if label.lower() in point[name].lower():
                preds.append(label.lower())
                break
        else: # for gemini
            pred = point[name].lower()
            if pred.startswith('**') and pred.endswith('** \n'):
                pred = pred[2:-4]  # Remove '**' from start and '** \n' from end
            elif pred.startswith('**') and pred.endswith('**'):
                pred = pred[2:-2]  # Remove '**' from start and end
            elif pred.endswith(' \n'):
                pred = pred[:-2]  # Remove ' \n' from end
            preds.append(pred)
        answers.append(point['answer'].lower())
    if task == "LaMP-3":
        preds.append(point[name])
        answers.append(point['answer'])
    if task == "LaMP-4":
        preds.append(point[name])
        answers.append(point['answer'])

print(len(preds))

metrics = compute_metrics(preds, answers)
print(metrics)

