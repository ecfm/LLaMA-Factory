"""
Script to convert CSV or Excel files to LLM input format.
The script takes an input file (CSV or Excel) and converts it to a JSON file
that can be used as input for LLM training.
Each review is split into individual sentences as separate inputs.
"""

import argparse
import json
import os

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize


# Download the necessary NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def convert_to_llm_format(input_file, output_file, column_name, min_len=3, max_len=256):
    """
    Convert a CSV or Excel file to LLM input format.
    Each review is split into individual sentences as separate inputs.
    
    Args:
        input_file (str): Path to the input file (CSV or Excel)
        output_file (str): Path to the output JSON file
        column_name (str): Name of the column containing the text data
        min_len (int): Minimum number of words in a sentence
        max_len (int): Maximum number of words in a sentence
    """
    # Get the prompt_id from the input filename (replace spaces with underscores)
    prompt_id = os.path.basename(input_file).split('.')[0].replace(' ', '_')

    # Determine file type and read accordingly
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    else:
        raise ValueError("Input file must be CSV or Excel format")

    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the input file")

    # Process the data
    all_data = []
    seen_sentences = set()
    ids = []
    uniq_sentences = []
    sentence_counter = 0

    for i, row in df.iterrows():
        review_text = str(row[column_name])

        # Skip empty reviews
        if not review_text.strip():
            continue

        # Split the review into sentences
        sentences = sent_tokenize(review_text)

        for sentence in sentences:
            # Skip sentences that are too short
            if len(sentence.split()) < min_len:
                continue

            # Skip sentences that are too long (to prevent Out-of-memory error)
            if len(sentence.split()) > max_len:
                continue

            # Remove duplicates
            if sentence in seen_sentences:
                continue

            seen_sentences.add(sentence)

            entry = {}
            entry['id'] = f"{prompt_id}_{sentence_counter}"
            ids.append(entry['id'])
            uniq_sentences.append(sentence)

            entry['conversations'] = [
                {
                    "from": "human",
                    "value": sentence
                },
                {
                    "from": "gpt",
                    "value": ''
                }
            ]
            all_data.append(entry)
            sentence_counter += 1

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the data as a JSON file
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)


    print(f"Processed {len(all_data)} sentences from the reviews")
    print(f"Sentence length filters: min={min_len}, max={max_len} words")
    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV or Excel file to LLM input format with sentence splitting')
    parser.add_argument('input_file', help='Path to the input file (CSV or Excel)')
    parser.add_argument('output_file', help='Path to the output JSON file')
    parser.add_argument('--column', required=True, help='Name of the column containing the text data')
    parser.add_argument('--min_len', type=int, default=5, help='Minimum number of words in a sentence (default: 5)')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum number of words in a sentence (default: 256)')

    args = parser.parse_args()

    convert_to_llm_format(args.input_file, args.output_file, args.column, args.min_len, args.max_len)


if __name__ == "__main__":
    main()
