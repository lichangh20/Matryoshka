import json
import os
import random
import requests
from tqdm import tqdm


def download_file(url, save_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the file with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f'Downloading {os.path.basename(save_path)}')
    
    # Save the file
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def download_lamp():
    for task in ["1", "3", "4"]:
        for split in ["train", "dev"]:
            print(f"\nDownloading LaMP-{task} {split} data...")
            url_questions = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{task}/{split}/{split}_questions.json"
            url_outputs = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{task}/{split}/{split}_outputs.json"
            save_path_questions = f"data/lamp/user/lamp{task}/{split}_questions.json"
            save_path_outputs = f"data/lamp/user/lamp{task}/{split}_outputs.json"
            download_file(url_questions, save_path_questions)
            download_file(url_outputs, save_path_outputs)
    for split in ["train", "dev"]:
        url_questions = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/{split}/{split}_questions.json"
        url_outputs = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/{split}/{split}_outputs.json"
        save_path_questions = f"data/lamp/user/lamp2n/{split}_questions.json"
        save_path_outputs = f"data/lamp/user/lamp2n/{split}_outputs.json"
        download_file(url_questions, save_path_questions)
        download_file(url_outputs, save_path_outputs)
        url_questions = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/new/{split}/{split}_questions.json"
        url_outputs = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/new/{split}/{split}_outputs.json"
        save_path_questions = f"data/lamp/user/lamp2m/{split}_questions.json"
        save_path_outputs = f"data/lamp/user/lamp2m/{split}_outputs.json"
        download_file(url_questions, save_path_questions)
        download_file(url_outputs, save_path_outputs)

def formalize_lamp():
    for split in ["train", "dev"]:
        for k in ["1", "2n", "2m", "3", "4"]:
            with open(f"data/lamp/user/lamp{k}/{split}_questions.json", "r") as f:
                questions = json.load(f)
            with open(f"data/lamp/user/lamp{k}/{split}_outputs.json", "r") as f:
                outputs = json.load(f)
            
            formalized_data = []
            for i in range(len(questions)):
                data = {
                    "question": questions[i]["input"],
                    "answer": outputs["golds"][i]["output"],
                    "id": questions[i]["id"],
                    "profile": questions[i]["profile"]
                }
                formalized_data.append(data)
            
            os.makedirs(f"data/lamp/formalized/lamp{k}/", exist_ok=True)
            with open(f"data/lamp/formalized/lamp{k}/formalized_{split}_data.jsonl", "w") as f:
                for item in formalized_data:
                    json.dump(item, f)
                    f.write("\n")

def observe_lamp():
    for split in ["train"]:
        for k in [4]:
            with open(f"data/lamp/formalized/lamp{k}/formalized_{split}_data.jsonl", "r") as f:
                data = [json.loads(line) for line in f]
            
            # random sample 10 data questions and answers
            sampled_data = random.sample(data, 1)
            for item in sampled_data:
                print("question:", item["question"], "\n\n")
                print("answer:", item["answer"], "\n\n")
                # print("profile:", item["profile"], "\n\n")
def main():
    download_lamp()
    formalize_lamp()
    # observe_lamp()

if __name__ == "__main__":
    main()