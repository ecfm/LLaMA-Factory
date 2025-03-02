import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from docx import Document
from split_text_util import split_into_paragraphs


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
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
    """Extract both full name and short name from filename."""
    name_part = re.sub(r'^\d+\.\s*', '', os.path.splitext(filename)[0])
    name_match = re.match(r'([^(]+)', name_part)
    if name_match:
        full_name = name_match.group(1).strip()
        # Get first word of name for matching shortened versions
        short_name = full_name.split()[0]
        return full_name, short_name
    raise DocumentProcessingError(f"Could not extract name from filename: {filename}")

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
    """Check if line contains speaker marker for the interviewee."""
    full_name, short_name = interviewee_names

    # Clean the line first but preserve key indicators
    clean_line = re.sub(r'\[\[.*?\]\]', '', line)  # Remove timestamps while keeping text
    clean_line = re.sub(r'\{.*?\}', '', clean_line)

    # Common speaker patterns found in transcripts
    patterns = [
        rf'^{full_name}\s*:',
        rf'^{short_name}\s*:',
        rf'^{full_name}\s*\n',
        rf'^{short_name}\s*\n',
        rf'^{full_name}\s*\([0-9:]+\)\s*:',  # Handle timestamp format (12:22)
        rf'^{short_name}\s*\([0-9:]+\)\s*:',
        rf'^{full_name}\s+\(', # Match name followed by timestamp
        # Add patterns specifically for various formats
        r'^Ken\s*:',
        r'^Ken\s*\n'
    ]

    # Case insensitive matching
    for pattern in patterns:
        if re.search(pattern, clean_line, re.IGNORECASE):
            return True

    return False


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
    """Find the most frequent speakers and identify interviewer/interviewee."""
    # Extract all speaker names
    speaker_pattern = re.compile(r'^([^:(\n]+)(?:\s*\(\d+:\d+\))?\s*:', re.MULTILINE)
    matches = speaker_pattern.findall(doc_content)

    # Count speaker frequencies
    speaker_counts = {}
    for speaker in matches:
        speaker = speaker.strip()
        speaker = re.sub(r'\s*\(\d+:\d+\)\s*$', '', speaker)  # Remove timestamps
        if speaker:
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

    if not speaker_counts:
        raise DocumentProcessingError("No speakers found in document")

    # Sort by frequency
    sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)
    top_speakers = [s[0] for s in sorted_speakers[:2]]  # Get top 2 speakers

    if len(top_speakers) < 2:
        raise DocumentProcessingError("Less than 2 speakers found in document")

    # Determine which is the interviewee based on filename match
    full_name, short_name = interviewee_names
    name_parts = set(full_name.lower().split()) | {short_name.lower()}

    # Calculate name similarity scores
    def name_similarity(speaker: str) -> float:
        speaker_parts = set(speaker.lower().split())
        return len(name_parts & speaker_parts) / max(len(name_parts), len(speaker_parts))

    similarities = [(s, name_similarity(s)) for s in top_speakers]
    similarities.sort(key=lambda x: x[1], reverse=True)

    interviewee = similarities[0][0]
    interviewer = [s for s in top_speakers if s != interviewee][0]

    return interviewee, interviewer

def is_interviewer_line(line: str, interviewer_name: str) -> bool:
    """Check if line is from the interviewer."""
    clean_line = clean_text(line)

    # Create patterns for interviewer name with optional timestamp
    patterns = [
        rf'^{interviewer_name}\s*:',
        rf'^{interviewer_name}\s*\(\d+:\d+\)\s*:',
        rf'^{interviewer_name}(?=\s|$)'
    ]

    return any(re.search(pattern, clean_line, re.IGNORECASE) for pattern in patterns)

def extract_interviewee_text(doc_content: str, interviewee_names: Tuple[str, str],
                           filename: str,  min_len: int, max_len: int, split_mode: str = 'sentence') -> Tuple[List[str], dict]:
    """Extract text with speaker detection."""
    if split_mode not in ['sentence', 'paragraph']:
        raise ValueError("split_mode must be either 'sentence' or 'paragraph'")

    # Find speakers
    interviewee, interviewer = find_speakers(doc_content, interviewee_names)
    logging.info(f"\nSpeakers detected in {filename}:")
    logging.info(f"  Interviewee: {interviewee}")
    logging.info(f"  Interviewer: {interviewer}")

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
            text = re.split(r':|(?=\()', line, 1)[-1].strip()
            if text:
                text = re.sub(r'\([0-9:]+\):', '', text)
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
    rejected_units = set()

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
            elif cleaned_unit not in rejected_units:
                rejected_units.add(cleaned_unit)
                logging.warning(f"REJECTED [{reason}] - {cleaned_unit}")

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
                    "value": "Analysis of this sentence would go here"
                }
            ]
        }
        json_data.append(entry)

    return json_data, id_mapping

def write_highlighted_transcript(filename: str, doc_content: str, id_mapping: dict,
                               output_dir: str, split_mode: str = 'sentence') -> None:
    """Write transcript with selected text units highlighted and numbered.
    
    Args:
        filename: Name of the original document
        doc_content: Full text content of the document
        id_mapping: Dictionary mapping cleaned text units to (original_text, unit_id) tuples
        output_dir: Directory to write highlighted transcripts
        split_mode: Whether processing by 'sentence' or 'paragraph'
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

    if unhighlighted_ids:
        logging.warning(f"\nWarning: {len(unhighlighted_ids)} text units were not highlighted in {filename}:")
        for unit, (original, unit_id) in id_mapping.items():
            if unit_id in unhighlighted_ids:
                logging.warning(f"ID {unit_id}: {original[:100]}...")
    else:
        logging.info(f"All {len(all_ids)} text units were successfully highlighted in {filename}")

    # Write output file
    output_file = os.path.join(output_dir, f"highlighted_{split_mode}s_{filename}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    return len(unhighlighted_ids)  # Return count of unhighlighted units for tracking

def process_directory(directory_path: str, output_file: str, highlights_dir: str, min_len: int, max_len: int,
                     split_mode: str = 'sentence') -> ProcessingStats:
    """Process all Word documents in the directory.
    
    Args:
        directory_path: Path to directory containing Word documents
        output_file: Path where JSON output should be written
        highlights_dir: Directory where highlighted transcripts should be saved
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

    # Create highlights directory if it doesn't exist
    if not os.path.exists(highlights_dir):
        os.makedirs(highlights_dir)
        logging.info(f"Created highlights directory: {highlights_dir}")

    # Get all Word documents
    doc_files = [f for f in os.listdir(directory_path)
                 if f.endswith('.docx') and not f.startswith('~$')]
    stats.total_files = len(doc_files)

    if stats.total_files == 0:
        logging.warning("No Word documents found in directory")
        return stats

    logging.info(f"\nProcessing {stats.total_files} documents...")
    logging.info("=" * 50)

    for filename in doc_files:
        file_path = os.path.join(directory_path, filename)
        logging.info(f"\nProcessing file: {filename}")

        # Extract names from filename
        try:
            interviewee_names = extract_name_from_filename(filename)
            logging.info(f"Extracted name: {interviewee_names[0]}")
        except DocumentProcessingError as e:
            logging.error(f"Error extracting name from {filename}: {str(e)}")
            stats.failed_files += 1
            continue

        # Read document content
        try:
            doc = Document(file_path)
            doc_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logging.error(f"Error reading file {filename}: {str(e)}")
            stats.failed_files += 1
            continue

        # Extract text units and their original forms
        try:
            text_units, unit_map = extract_interviewee_text(
                doc_content,
                interviewee_names,
                filename,
                min_len,
                max_len,
                split_mode
            )
        except Exception as e:
            logging.error(f"Error extracting text from {filename}: {str(e)}")
            stats.failed_files += 1
            continue

        # Track units for this file
        all_units[filename] = len(text_units)

        # Validate minimum number of units
        if len(text_units) < 3:
            logging.warning(
                f"File {filename}: Found only {len(text_units)} {split_mode}s "
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

                logging.info(
                    f"Successfully processed {filename}: "
                    f"{len(text_units)} {split_mode}s extracted"
                )

                # Write highlighted transcript using the same ID mapping
                write_highlighted_transcript(
                    filename,
                    doc_content,
                    id_mapping,
                    highlights_dir,
                    split_mode
                )
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                stats.failed_files += 1
                continue
        else:
            logging.error(f"No valid text units found in file: {filename}")
            stats.failed_files += 1

    # Log summary statistics
    logging.info("\nProcessing Summary:")
    logging.info("=" * 50)
    for filename, count in all_units.items():
        logging.info(f"{filename}: {count} {split_mode}s")
    logging.info("-" * 50)
    logging.info(f"Total files processed: {stats.processed_files}")
    logging.info(f"Total {split_mode}s extracted: {stats.total_sentences}")
    logging.info(f"Files with warnings: {stats.files_with_warnings}")
    logging.info(f"Failed files: {stats.failed_files}")
    logging.info("=" * 50)

    # Write the final JSON file if we have data
    if all_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully wrote {len(all_data)} entries to {output_file}")
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
    parser.add_argument('--highlights_dir', type=str, required=True,
                        help='Directory where highlighted transcripts should be saved')
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
        highlights_dir=args.highlights_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        split_mode=args.split_mode
    )
