import pandas as pd
import json

# Load the existing CSV file
csv_path = "/Users/maoc/MIT Dropbox/Chengfeng Mao/GPT VOC/LLM finetuning/GPT_S2N_data/shuffled_answers/all_13b.csv"
df = pd.read_csv(csv_path)

# Load the inference results JSON file
json_path = "/Users/maoc/Downloads/inference_results(1).json"
with open(json_path, 'r') as f:
    inference_results = json.load(f)

# Create a dictionary mapping id to output from inference results
inference_dict = {item['id']: item['output'] for item in inference_results}

# Add a new column for Finetuned Qwen 14b
df['Finetuned Qwen 14b'] = df['id'].map(inference_dict)

# Save the updated dataframe to a new CSV file
output_path = "/Users/maoc/MIT Dropbox/Chengfeng Mao/GPT VOC/LLM finetuning/GPT_S2N_data/shuffled_answers/all_13b_with_qwen.csv"
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to {output_path}")