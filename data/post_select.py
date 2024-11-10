import sys
import json
import os
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")    
HF_TOKEN="your_huggingface_token"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.classification import create_metric_f1_accuracy_chatgpt, create_metric_mae_rmse_chatgpt
from metrics.generation import create_metric_bleu_rouge_meteor_chatgpt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#======================= The params to change for different tasks =======================
task = 'LaMP-1'
index = task.split('-')[1].lower()
file_dir = f"data/lamp/generated/lamp{index}"
file_path = f"{file_dir}/{task}.json"
#=======================================================================================


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


with open(file_path, 'r') as f:
    data = json.load(f)

processed_data = []
for point in tqdm(data):
    processed_point = point.copy()  # Create a copy of the original data point
    true_answer = point['answer']
    true_labels = [true_answer]
    
    # Calculate metrics for each prediction and store with original index
    scored_preds = []
    for idx, pred in enumerate(point['positive']):
        if pred['response'] is None:
            continue
        pred_label = [pred['response']]
        metrics = compute_metrics(true_labels, pred_label)
        scored_preds.append((idx, metrics, pred.copy()))  # Use a copy of the prediction
    
    # Sort predictions based on the main metric score (assuming it's the first one)
    if task == 'LaMP-3':
        sorted_preds = sorted(scored_preds, key=lambda x: list(x[1].values())[0], reverse=False)
    else:
        sorted_preds = sorted(scored_preds, key=lambda x: list(x[1].values())[0], reverse=True)
    
    # Calculate the midpoint
    midpoint = len(sorted_preds) // 2

    # Update the 'positive' and 'negative' lists in the processed point
    processed_point['positive'] = [pred for _, _, pred in sorted_preds[:midpoint]]
    processed_point['negative'] = [pred for _, _, pred in sorted_preds[midpoint:]]

    # Remove the 'metrics' and 'index' keys from each prediction in 'positive' and 'negative'
    for pred in processed_point['positive'] + processed_point['negative']:
        pred.pop('metrics', None)
        pred.pop('index', None)

    processed_data.append(processed_point)

# After processing all data points, write the processed data to a new JSON file
output_file_path = file_path.replace('.json', '_processed.json')
with open(output_file_path, 'w') as f:
    json.dump(processed_data, f, indent=2)




