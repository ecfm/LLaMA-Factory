import json
import os

def convert_jsonl_to_json(input_file, output_file):
    """
    Convert a JSONL file to a JSON array format without images.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSON file
    """
    # List to store all conversation objects
    conversations = []
    
    # Read the JSONL file line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as a JSON object
            conversation = json.loads(line.strip())
            
            # Extract only the messages part (without images)
            if 'messages' in conversation:
                conversations.append({
                    'messages': conversation['messages']
                })
    
    # Write the conversations to the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Converted {len(conversations)} conversations.")

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "data/ft_risky_AB.jsonl"
    output_file = "data/ft_risky_AB_converted.json"
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Perform the conversion
    convert_jsonl_to_json(input_file, output_file) 