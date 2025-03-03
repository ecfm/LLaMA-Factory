import argparse
import json
import logging
import os
import re
import warnings
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Suppress huggingface warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Global model instance to avoid reloading
_model_instance = None

def get_sentence_transformer_model():
    """Get or create a singleton instance of the SentenceTransformer model."""
    global _model_instance
    if _model_instance is None:
        logging.info("Loading SentenceTransformer model (one-time initialization)")
        _model_instance = SentenceTransformer('all-MiniLM-L6-v2')
    return _model_instance

def split_into_paragraphs(text, min_len=100, max_len=200):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)

    # Compute word counts for each sentence
    sentence_word_counts = [len(word_tokenize(sentence)) for sentence in sentences]

    # Compute cumulative word counts
    cumulative_word_counts = [0]
    for wc in sentence_word_counts:
        cumulative_word_counts.append(cumulative_word_counts[-1] + wc)

    # Use the singleton model instance
    model = get_sentence_transformer_model()

    # Compute embeddings for all sentences
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embeddings = model.encode(sentences, show_progress_bar=False)

    # Compute similarities between consecutive sentences
    similarities = [0]  # Similarity for the first sentence is zero
    for i in range(1, num_sentences):
        sim = cosine_similarity(
            [embeddings[i-1]],
            [embeddings[i]]
        )[0][0]
        similarities.append(sim)

    # Compute cumulative similarities
    cumulative_similarities = [0]
    for sim in similarities[1:]:
        cumulative_similarities.append(cumulative_similarities[-1] + sim)

    # Initialize dynamic programming arrays
    dp = [float('-inf')] * (num_sentences + 1)  # dp[i] stores max total similarity up to sentence i
    dp[0] = 0  # Base case: no sentences means zero similarity

    # back_pointers[i] stores the index j where the optimal split before position i occurs
    back_pointers = [-1] * (num_sentences + 1)

    # Dynamic programming to compute dp[i]
    for i in range(1, num_sentences + 1):
        for j in range(0, i):
            # Word count between sentences j to i-1
            word_count = cumulative_word_counts[i] - cumulative_word_counts[j]
            if min_len <= word_count <= max_len:
                # Similarity score for paragraph from sentence j to i-1
                paragraph_similarity = cumulative_similarities[i-1] - cumulative_similarities[j]
                total_similarity = dp[j] + paragraph_similarity
                if total_similarity > dp[i]:
                    dp[i] = total_similarity
                    back_pointers[i] = j

    # Reconstruct paragraphs using back_pointers
    paragraphs = []
    idx = num_sentences
    indices = []
    while idx > 0:
        indices.append(idx)
        idx = back_pointers[idx]
    indices.append(0)
    indices = indices[::-1]

    for k in range(len(indices) - 1):
        start = indices[k]
        end = indices[k+1]
        paragraph_sentences = sentences[start:end]
        paragraphs.append(' '.join(paragraph_sentences))

    return paragraphs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interview_processing.log'),
        logging.StreamHandler()
    ]
)

# Add debug log file for sentence filtering
debug_log = logging.FileHandler('sentence_filtering.log')
debug_log.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(message)s')
debug_log.setFormatter(debug_formatter)
logging.getLogger().addHandler(debug_log)

# Configure specific loggers to reduce noise
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('filelock').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

@dataclass
class ProcessingStats:
    """Statistics for document processing"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_sentences: int = 0
    files_with_warnings: int = 0

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

def extract_name_from_filename(filename: str) -> Tuple[str, str]:
    """Return fixed participant name for all files."""
    # We're using fixed labels, so we don't need to extract names from filenames
    return "Participant", "Interviewer"

def clean_text(text: str) -> str:
    """Clean transcript text of artifacts while preserving sentence structure."""

    # Preserve quoted speech
    text = re.sub(r'"\s*([^"]+)\s*"', r' "\1" ', text)
    # Handle timestamps and formatting
    text = re.sub(r'\[\[\d{2}:\d{2}\]\]', ' ', text)  # Convert timestamps to spaces
    text = re.sub(r'\(\d{2}:\d{2}\):', ' ', text)  # Convert timestamps to spaces
    text = re.sub(r'\[inaudible[^\]]*\]', '', text)
    text = re.sub(r'\{\.[^}]+\}', '', text)  # Remove formatting
    text = re.sub(r'\[[^\]]+\]\([^)]+\)', '', text)  # Remove links
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
    text = re.sub(r'\[inaudible \d{2}:\d{2}:\d{2}\]', '', text)

    # Fix quotes and apostrophes
    text = text.replace('\\\'', "'")
    text = re.sub(r'"{2,}', '"', text)

    # Clean whitespace but preserve sentence boundaries
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)  # Normalize period spacing
    text = re.sub(r'\s*\?\s*', '? ', text)
    text = re.sub(r'\s*!\s*', '! ', text)

    return text.strip()

def is_speaker_line(line: str, interviewee_names: Tuple[str, str]) -> bool:
    """Check if line contains speaker marker for the interviewee (Participant)."""
    # Clean the line first but preserve key indicators
    clean_line = re.sub(r'\[\[.*?\]\]', '', line)  # Remove timestamps while keeping text
    clean_line = re.sub(r'\{.*?\}', '', clean_line)
    
    # Fixed pattern for participant
    return re.search(r'^Participant\s*:', clean_line, re.IGNORECASE) is not None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences with sophisticated rules."""
    text = clean_text(text)
    sentences = []

    # Split on obvious sentence boundaries
    chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    for chunk in chunks:
        if not chunk.strip():
            continue

        # Check if chunk looks like a complete sentence
        if re.search(r'[.!?]$', chunk.strip()):
            sentences.append(chunk.strip())
        else:
            # Check for sentence-like structures without punctuation
            if re.search(r'\b(but|and|or|so)\s*$', chunk):
                logging.debug(f"FILTERED in split: Fragment ending with conjunction - '{chunk}'")
                continue
            if len(chunk.split()) > 3:  # If substantial enough, treat as sentence
                sentences.append(chunk.strip() + '.')
                # logging.debug(f"Added period to: '{chunk}'")

    return sentences

def validate_text_unit(text: str, split_mode: str = 'sentence') -> Tuple[bool, Optional[str]]:
    """Validate a text unit (sentence or paragraph) with consistent rules."""
    text = clean_text(text)

    # Common validation for both modes
    if not re.search(r'[a-zA-Z]', text):
        return False, "No letters found"

    if len(text.split()) < 5:
        return False, "Too few words"
    if len(text) < 15:
        return False, "Text too short"

    return True, None


def find_speakers(doc_content: str, interviewee_names: Tuple[str, str]) -> Tuple[str, str]:
    """Return fixed speaker labels."""
    # Fixed speaker labels
    return "Participant", "Interviewer"

def is_interviewer_line(line: str, interviewer_name: str) -> bool:
    """Check if line is from the interviewer."""
    clean_line = re.sub(r'\[\[.*?\]\]', '', line)  # Remove timestamps while keeping text
    clean_line = re.sub(r'\{.*?\}', '', clean_line)
    
    # Fixed pattern for interviewer
    return re.search(r'^Interviewer\s*:', clean_line, re.IGNORECASE) is not None

def extract_interviewee_text(doc_content: str, interviewee_names: Tuple[str, str],
                           filename: str,  min_len: int, max_len: int, split_mode: str = 'sentence') -> Tuple[List[str], dict]:
    """Extract text with speaker detection."""
    if split_mode not in ['sentence', 'paragraph']:
        raise ValueError("split_mode must be either 'sentence' or 'paragraph'")

    # Get fixed speaker labels
    interviewee, interviewer = find_speakers(doc_content, interviewee_names)
    logging.info("\nSpeakers detected:")
    logging.info(f"  • Interviewee: {interviewee}")
    logging.info(f"  • Interviewer: {interviewer}")

    # Phase 1: Collect raw text segments
    raw_segments = []
    current_segment = []
    current_speaker = None
    lines = doc_content.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if current_speaker == 'interviewee' and current_segment:
                raw_segments.append(' '.join(current_segment))
                current_segment = []
            continue

        if is_speaker_line(line, interviewee_names):
            if current_speaker == 'interviewee' and current_segment:
                raw_segments.append(' '.join(current_segment))
                current_segment = []

            current_speaker = 'interviewee'
            # Get text after speaker marker
            text = re.split(r':', line, 1)[-1].strip()
            if text:
                current_segment.append(text.strip())

        elif is_interviewer_line(line, interviewer):
            if current_speaker == 'interviewee' and current_segment:
                raw_segments.append(' '.join(current_segment))
                current_segment = []
            current_speaker = 'interviewer'

        elif current_speaker == 'interviewee':
            # Check for paragraph breaks
            if (i > 0 and not lines[i-1].strip()) or re.match(r'^\s+', lines[i]):
                if current_segment:
                    raw_segments.append(' '.join(current_segment))
                    current_segment = []

            if line and not line.isspace():
                current_segment.append(line)

    # Add final segment if exists
    if current_speaker == 'interviewee' and current_segment:
        raw_segments.append(' '.join(current_segment))

    # Phase 2: Join continued segments
    joined_segments = []
    pending_segment = None

    for segment in raw_segments:
        if pending_segment:
            segment = f"{pending_segment} {segment}"
            pending_segment = None

        if segment.strip().endswith('...'):
            pending_segment = segment
            continue

        if segment.strip().startswith('-') and joined_segments:
            # Join with previous segment
            prev_segment = joined_segments.pop()
            segment = f"{prev_segment} {segment.lstrip('-').strip()}"

        joined_segments.append(segment)

    # Add final pending segment if exists
    if pending_segment:
        joined_segments.append(pending_segment)


    # Phase 3: Clean, split, and validate
    text_units = []
    unit_map = {}
    rejected_units = {}  # Dictionary to group rejected units by reason

    for segment in joined_segments:
        # Store original before any cleaning
        original_segment = segment
        cleaned_segment = clean_text(segment)
        if not cleaned_segment:
            continue

        # Split based on mode
        if split_mode == 'sentence':
            # Track the position in original text while splitting
            current_pos = 0
            units = []
            for cleaned_unit in split_into_sentences(cleaned_segment):
                # Find this sentence in the original text
                sentence_start = original_segment.find(cleaned_unit.rstrip('.!?'), current_pos)
                if sentence_start != -1:
                    # Find the end of the sentence in original text
                    sentence_end = sentence_start + len(cleaned_unit)
                    original_unit = original_segment[sentence_start:sentence_end].strip()
                    units.append((cleaned_unit, original_unit))
                    current_pos = sentence_end
                else:
                    # Fallback if exact match not found
                    units.append((cleaned_unit, cleaned_unit))
        else:  # paragraph mode
            if len(cleaned_segment.split()) > max_len:
                # Use the singleton model for paragraph splitting
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    units = [(unit, unit) for unit in split_into_paragraphs(cleaned_segment, min_len, max_len)]
                logging.debug(f"SPLIT text > {max_len} words into {len(units)} units")
            else:
                units = [(cleaned_segment, original_segment)]

        # Validate and store units
        for cleaned_unit, original_unit in units:
            if not cleaned_unit:
                continue

            # Ensure proper ending punctuation
            if not re.search(r'[.!?]$', cleaned_unit):
                cleaned_unit += '.'
                original_unit = original_unit.rstrip() + '.'

            is_valid, reason = validate_text_unit(cleaned_unit, split_mode)
            if is_valid:
                text_units.append(cleaned_unit)
                unit_map[cleaned_unit] = original_unit
            else:
                # Group rejected units by reason
                if reason not in rejected_units:
                    rejected_units[reason] = set()  # Use a set to store unique units
                rejected_units[reason].add(cleaned_unit)

    # Log rejected units grouped by reason in a compact format
    if rejected_units:
        logging.info("\nOmitted text units:")
        for reason, units_set in rejected_units.items():
            # Convert set to list and get unique examples (up to 3)
            unique_examples = list(units_set)[:3]
            example_count = len(unique_examples)
            total_count = len(units_set)

            # Format examples as a numbered list on a single line
            examples_text = "; ".join([f"{i+1}) {ex}" for i, ex in enumerate(unique_examples)])

            # Add ellipsis if there are more examples
            if total_count > example_count:
                examples_text += f"; ... ({total_count - example_count} more)"

            logging.info(f"  • [{reason}] - {total_count} units: {examples_text}")

    logging.debug(f"\nTotal {split_mode}s extracted: {len(text_units)}")
    logging.debug("="*50 + "\n")

    return text_units, unit_map

def create_id_mapping(text_units: List[str], interviewee_name: str, unit_map: dict) -> Tuple[List[dict], dict]:
    """Create consistent IDs for both JSON and highlights."""
    json_data = []
    id_mapping = {}  # Map text units to their IDs

    # Replace spaces with underscores for the ID
    id_name = interviewee_name.replace(' ', '_')

    for i, unit in enumerate(text_units, 1):
        unit_id = f"{id_name}_{i}"
        original_text = unit_map.get(unit, unit)

        # Store mapping for highlighting
        id_mapping[unit] = (original_text, unit_id)

        # Create JSON entry
        entry = {
            "id": unit_id,
            "conversations": [
                {
                    "from": "human",
                    "value": f'<GPT-VOC> <PRODUCT_CATEGORY="conference"> \n[REVIEW]::{unit}'
                },
                {
                    "from": "gpt",
                    "value": ""
                }
            ]
        }
        json_data.append(entry)

    return json_data, id_mapping

def write_highlighted_transcript(filename: str, doc_content: str, id_mapping: dict,
                               output_dir: str, split_mode: str = 'sentence') -> int:
    """Write transcript with selected text units highlighted and numbered.
    
    Args:
        filename: Name of the original document
        doc_content: Full text content of the document
        id_mapping: Dictionary mapping cleaned text units to (original_text, unit_id) tuples
        output_dir: Directory to write highlighted transcripts
        split_mode: Whether processing by 'sentence' or 'paragraph'
        
    Returns:
        Count of unhighlighted units
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_lines = []
    highlighted_ids = set()  # Track which IDs have been highlighted

    # Create a mapping of original text to IDs for easier lookup
    original_to_id = {}
    for unit, (original, unit_id) in id_mapping.items():
        # Store both exact and normalized versions for matching
        original_to_id[original.strip()] = unit_id
        original_to_id[clean_text(original).strip()] = unit_id

    # Process each line
    for line in doc_content.split('\n'):
        line_stripped = line.strip()
        cleaned_line = clean_text(line).strip()

        # Try matching both original and cleaned versions
        unit_id = original_to_id.get(line_stripped) or original_to_id.get(cleaned_line)

        if unit_id:
            output_lines.append(f">>> [{unit_id}] {line} <<<")
            highlighted_ids.add(unit_id)
        else:
            output_lines.append(line)

    # Check for unhighlighted text units
    all_ids = {unit_id for _, (_, unit_id) in id_mapping.items()}
    unhighlighted_ids = all_ids - highlighted_ids

    # Write output file
    output_file = os.path.join(output_dir, f"highlighted_{split_mode}s_{filename}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    return len(unhighlighted_ids)  # Return count of unhighlighted units for tracking

def process_directory(directory_path: str, output_file: str, highlight_dir: Optional[str], min_len: int, max_len: int,
                     split_mode: str = 'sentence') -> ProcessingStats:
    """Process all Word documents in the directory.
    
    Args:
        directory_path: Path to directory containing Word documents
        output_file: Path where JSON output should be written
        highlight_dir: Optional directory where highlighted transcripts should be saved for debugging
                       If None, no highlighted transcripts will be generated
        min_len: Minimum length of text units in words
        max_len: Maximum length of text units in words
        split_mode: Whether to process by 'sentence' or 'paragraph'
        
    Returns:
        ProcessingStats object containing processing statistics
        
    Raises:
        FileNotFoundError: If directory_path doesn't exist
    """
    stats = ProcessingStats()
    all_data = []
    all_units = {}  # Track text units per file

    # Validate directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Create highlights directory if specified and doesn't exist
    if highlight_dir and not os.path.exists(highlight_dir):
        os.makedirs(highlight_dir)
        logging.info(f"Created highlights directory: {highlight_dir}")

    # Get all Word documents
    doc_files = [f for f in os.listdir(directory_path)
                 if f.endswith('.docx') and not f.startswith('~$')]
    stats.total_files = len(doc_files)

    if stats.total_files == 0:
        logging.warning("No Word documents found in directory")
        return stats

    # Pre-load the model once before processing any files
    _ = get_sentence_transformer_model()

    # Print initial header before progress bar starts
    print("\n" + "="*70)
    print(f"STARTING BATCH PROCESSING: {stats.total_files} documents")
    print("="*70 + "\n")

    # Create progress bar with position=0 to allow other progress bars to appear below it
    progress_bar = tqdm(doc_files, desc="Processing files", unit="file", position=0, leave=True)

    # Process each file
    for filename in progress_bar:
        file_path = os.path.join(directory_path, filename)
        progress_bar.set_description(f"Processing {filename}")

        # Collect log messages for this file to display after progress bar updates
        file_logs = []
        file_logs.append("\n" + "-"*70)
        file_logs.append(f"DOCUMENT: {filename}")
        file_logs.append("-"*70)

        # Get fixed participant name
        interviewee_names = extract_name_from_filename(filename)
        file_logs.append(f"Using fixed speaker labels: {interviewee_names[0]} and {interviewee_names[1]}")

        # Read document content
        try:
            doc = Document(file_path)
            doc_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            file_logs.append(f"ERROR: Error reading file {filename}: {str(e)}")
            stats.failed_files += 1
            continue

        # Extract text units and their original forms
        try:
            # Temporarily redirect logging to our collection
            text_units, unit_map = extract_interviewee_text(
                doc_content,
                interviewee_names,
                filename,
                min_len,
                max_len,
                split_mode
            )
        except Exception as e:
            file_logs.append(f"ERROR: Error extracting text from {filename}: {str(e)}")
            stats.failed_files += 1
            continue

        # Track units for this file
        all_units[filename] = len(text_units)

        # Validate minimum number of units
        if len(text_units) < 3:
            file_logs.append(
                f"WARNING: File {filename}: Found only {len(text_units)} {split_mode}s "
                "(minimum 3 expected)"
            )
            stats.files_with_warnings += 1

        if text_units:
            try:
                # Create consistent IDs for both JSON and highlights
                json_data, id_mapping = create_id_mapping(
                    text_units,
                    interviewee_names[0],
                    unit_map
                )

                # Add to overall data
                all_data.extend(json_data)
                stats.total_sentences += len(text_units)
                stats.processed_files += 1

                # Update progress bar with additional information
                progress_bar.set_postfix(units=len(text_units), total_units=stats.total_sentences)

                file_logs.append(f"✓ Successfully processed: {len(text_units)} {split_mode}s extracted")

                # Write highlighted transcript if highlight_dir is specified
                if highlight_dir:
                    unhighlighted_count = write_highlighted_transcript(
                        filename,
                        doc_content,
                        id_mapping,
                        highlight_dir,
                        split_mode
                    )
                    if unhighlighted_count > 0:
                        file_logs.append(f"WARNING: {unhighlighted_count} text units could not be highlighted")
            except Exception as e:
                file_logs.append(f"ERROR: Error processing {filename}: {str(e)}")
                stats.failed_files += 1
                continue
        else:
            file_logs.append(f"ERROR: No valid text units found in file: {filename}")
            stats.failed_files += 1

        # Pause the progress bar, print logs, then resume
        progress_bar.clear()
        for log in file_logs:
            if log.startswith("ERROR:"):
                logging.error(log[7:])  # Remove the "ERROR: " prefix
            elif log.startswith("WARNING:"):
                logging.warning(log[9:])  # Remove the "WARNING: " prefix
            else:
                logging.info(log)
        progress_bar.display()

    # Close the progress bar
    progress_bar.close()

    # Log summary statistics
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)

    print("\nExtracted units by file:")
    for filename, count in all_units.items():
        print(f"  • {filename}: {count} {split_mode}s")

    print("\nOverall statistics:")
    print(f"  • Total files processed: {stats.processed_files}/{stats.total_files}")
    print(f"  • Total {split_mode}s extracted: {stats.total_sentences}")
    print(f"  • Files with warnings: {stats.files_with_warnings}")
    print(f"  • Failed files: {stats.failed_files}")
    print("="*70)

    # Write the final JSON file if we have data
    if all_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Successfully wrote {len(all_data)} entries to {output_file}")
        except Exception as e:
            logging.error(f"Error writing output file: {str(e)}")
            raise
    else:
        logging.error("No valid data to write to output file")

    return stats

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process interview transcripts for LLM input.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing Word documents to process')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path where JSON output should be written')
    parser.add_argument('--highlight_dir', type=str, required=False, default=None,
                        help='Optional directory to save highlighted transcripts. These transcripts show which parts of the original text were extracted and their IDs.')
    parser.add_argument('--min_len', type=int, default=30,
                        help='Minimum length of text units in words (default: 30)')
    parser.add_argument('--max_len', type=int, default=80,
                        help='Maximum length of text units in words (default: 80)')
    parser.add_argument('--split_mode', type=str, choices=['sentence', 'paragraph'], default='paragraph',
                        help='Whether to process by sentence or paragraph (default: paragraph)')

    # Parse arguments
    args = parser.parse_args()

    # Process directory with provided arguments
    stats = process_directory(
        directory_path=args.input_dir,
        output_file=args.output_file,
        highlight_dir=args.highlight_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        split_mode=args.split_mode
    )
