import argparse
import json
import os
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Check if NLTK resources are already downloaded before downloading
def download_nltk_resources():
    nltk_data_path = nltk.data.path

    # Check for punkt
    if not os.path.exists(os.path.join(nltk_data_path[0], 'tokenizers', 'punkt')):
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)

    # Check for stopwords
    if not os.path.exists(os.path.join(nltk_data_path[0], 'corpora', 'stopwords')):
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

    # Check for wordnet
    if not os.path.exists(os.path.join(nltk_data_path[0], 'corpora', 'wordnet')):
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=True)

# Download required NLTK resources if needed
download_nltk_resources()

def extract_review_text(text):
    """Extract the review text after [REVIEW]:: pattern"""
    match = re.search(r'\[REVIEW\]::(.*)', text)
    if match:
        return match.group(1).strip()
    return text

def normalize_text(text):
    """Normalize text by converting to lowercase, removing punctuation, and lemmatizing"""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(lemmatized_words)

def generate_keywords_tfidf(all_texts, text_index, num_keywords=3):
    """Generate keywords using TF-IDF to identify the most important terms"""
    # Normalize all texts
    normalized_texts = [normalize_text(text) for text in all_texts]

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.9,
        stop_words=stopwords.words('english'),
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with at least 3 characters
    )

    # Fit and transform the texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(normalized_texts)

    # Get feature names (terms)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get the TF-IDF scores for the current text
    tfidf_scores = tfidf_matrix[text_index].toarray()[0]

    # Create a dictionary of term:score
    term_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}

    # Sort terms by score and get the top ones
    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)

    # Extract the top keywords
    keywords = [term for term, score in sorted_terms[:num_keywords]]

    # Ensure we have exactly num_keywords
    while len(keywords) < num_keywords:
        keywords.append('')

    return keywords[:num_keywords]

def process_data(input_file, output_file):
    print(f"Processing data from {input_file} to {output_file}...")

    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create a list to store the extracted data
    extracted_data = []

    # First, collect all output texts for TF-IDF processing
    all_outputs = []
    for item in data:
        output_text = item.get('output', '')
        all_outputs.append(output_text)

    # Process each item
    for i, item in enumerate(data):
        id_value = item.get('id', '')

        # Extract input text (review part)
        input_text = extract_review_text(item.get('input', ''))

        # Get output and target
        output_text = item.get('output', '')

        # Generate keywords using TF-IDF
        keywords = generate_keywords_tfidf(all_outputs, i)

        # Add to our data list
        extracted_data.append({
            'id': id_value,
            'input': input_text,
            'result': output_text,
            'keyword_1': keywords[0],
            'keyword_2': keywords[1],
            'keyword_3': keywords[2]
        })

    # Create a DataFrame
    df = pd.DataFrame(extracted_data)

    # Create an CSV writer
    df.to_csv(output_file, index=False)

    print(f"Data processed and saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process VOC data from JSON to CSV with keywords.')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('output_file', help='Path to the output CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Process the data
    process_data(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
