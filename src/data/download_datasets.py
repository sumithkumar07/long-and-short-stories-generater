import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import gutenbergpy.textget
import pandas as pd
import requests
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_writing_prompts(self) -> None:
        """Download the WritingPrompts dataset from Hugging Face."""
        logger.info("WritingPrompts dataset is not available. Skipping download.")

    def download_rocstories(self) -> None:
        """Download the ROCStories dataset (manual download required)."""
        logger.info("ROCStories dataset requires manual download due to licensing. Please follow the instructions at: https://huggingface.co/datasets/story_cloze and place the data in the appropriate directory.")
        # Skipping automatic download

    def download_hellaswag(self) -> None:
        """Download the hellaswag dataset as an alternative to ROCStories."""
        logger.info("Downloading hellaswag dataset...")
        try:
            dataset = load_dataset("hellaswag", trust_remote_code=True)
            for split in dataset.keys():
                output_file = self.raw_dir / f"hellaswag_{split}.json"
                dataset[split].to_json(output_file)
            logger.info("hellaswag dataset downloaded successfully!")
        except Exception as e:
            logger.error(f"Error downloading hellaswag dataset: {str(e)}")

    def download_gutenberg_books(self, book_ids=None) -> None:
        """Download books from Project Gutenberg using a fixed list of book IDs."""
        logger.info(f"Downloading books from Project Gutenberg...")
        # Example English book IDs (public domain)
        if book_ids is None:
            book_ids = [1342, 84, 11, 1661, 2701]  # Pride and Prejudice, Frankenstein, Alice, Sherlock Holmes, Moby Dick
        try:
            for book_id in tqdm(book_ids):
                try:
                    text = gutenbergpy.textget.get_text_by_id(book_id)
                    if text:
                        clean_text = gutenbergpy.textget.strip_headers(text)
                        output_file = self.raw_dir / f"gutenberg_{book_id}.txt"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(clean_text.decode('utf-8'))
                except Exception as e:
                    logger.warning(f"Error downloading book {book_id}: {str(e)}")
                    continue
            logger.info("Project Gutenberg books downloaded successfully!")
        except Exception as e:
            logger.error(f"Error downloading Project Gutenberg books: {str(e)}")

    def download_short_story_archive(self) -> None:
        """Placeholder for downloading Short Story Archive dataset."""
        logger.info("Short Story Archive dataset download not implemented. Please add manually if available.")

    def download_fanfiction(self) -> None:
        """Placeholder for downloading Fanfiction datasets (AO3, Fanfiction.net)."""
        logger.info("Fanfiction dataset download not implemented. Please add manually if available.")

    def download_hf_dataset(self, dataset_name, tag, splits=None, trust_remote_code=False):
        """Generic function to download a Hugging Face dataset and save splits."""
        logger.info(f"Downloading {dataset_name} dataset...")
        try:
            dataset = load_dataset(dataset_name, trust_remote_code=trust_remote_code)
            for split in (splits or dataset.keys()):
                output_file = self.raw_dir / f"{dataset_name.replace('/', '_')}_{split}.json"
                dataset[split].to_json(output_file)
            logger.info(f"{dataset_name} dataset downloaded successfully!")
        except Exception as e:
            logger.error(f"Error downloading {dataset_name} dataset: {str(e)}")

    def download_all_alternatives(self):
        # Short story datasets
        self.download_hf_dataset('aeslc', '[SHORT]')
        self.download_hf_dataset('xsum', '[SHORT]', trust_remote_code=True)
        # Long story datasets
        self.download_hf_dataset('gutenberg_time', '[LONG]')
        # Removed: GEM/wiki_auto (not available), bookcorpusopen, pg19, multi_news

    def process_datasets(self) -> None:
        """Process the downloaded datasets into a unified format with [SHORT] or [LONG] tags."""
        logger.info("Processing datasets...")
        
        # Process WritingPrompts
        try:
            for split in ['train', 'validation', 'test']:
                input_file = self.raw_dir / f"writing_prompts_{split}.json"
                if input_file.exists():
                    df = pd.read_json(input_file)
                    processed_data = []
                    for _, row in df.iterrows():
                        processed_data.append({
                            'prompt': f"[SHORT] {row['prompt']}",
                            'story': f"[SHORT] {row['story']}",
                            'type': 'short',
                            'source': 'writing_prompts'
                        })
                    output_file = self.processed_dir / f"processed_writing_prompts_{split}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error processing WritingPrompts dataset: {str(e)}")

        # Process Gutenberg books
        try:
            gutenberg_files = list(self.raw_dir.glob('gutenberg_*.txt'))
            processed_books = []
            for file in tqdm(gutenberg_files):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    sections = text.split('\n\n')
                    for section in sections:
                        if len(section.strip()) > 100:
                            processed_books.append({
                                'text': f"[LONG] {section.strip()}",
                                'type': 'long',
                                'source': 'gutenberg',
                                'book_id': file.stem.split('_')[1]
                            })
                except Exception as e:
                    logger.warning(f"Error processing book {file}: {str(e)}")
                    continue
            output_file = self.processed_dir / "processed_gutenberg.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_books, f, indent=2)
        except Exception as e:
            logger.error(f"Error processing Gutenberg books: {str(e)}")

        # Placeholder for processing Short Story Archive and Fanfiction datasets
        logger.info("Processing for Short Story Archive and Fanfiction datasets not implemented.")

    def process_alternative_datasets(self):
        """Process the alternative datasets and add [SHORT] or [LONG] tags."""
        # Short story datasets
        short_datasets = [
            ('aeslc', 'email_body', 'subject_line'),
            ('xsum', 'document', 'summary'),
        ]
        for ds_name, prompt_field, story_field in short_datasets:
            for split in ['train', 'validation', 'test']:
                input_file = self.raw_dir / f"{ds_name}_{split}.json"
                if input_file.exists():
                    try:
                        df = pd.read_json(input_file, lines=True)
                        processed_data = []
                        for _, row in df.iterrows():
                            prompt = row.get(prompt_field, '')
                            story = row.get(story_field, '')
                            processed_data.append({
                                'prompt': f"[SHORT] {prompt}",
                                'story': f"[SHORT] {story}",
                                'type': 'short',
                                'source': ds_name
                            })
                        output_file = self.processed_dir / f"processed_{ds_name}_{split}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(processed_data, f, indent=2)
                    except Exception as e:
                        logger.error(f"Error processing {input_file}: {str(e)}")
        # Long story datasets
        long_datasets = [
            ('gutenberg_time', 'text'),
        ]
        for ds_name, story_field in long_datasets:
            for split in ['train', 'validation', 'test']:
                input_file = self.raw_dir / f"{ds_name}_{split}.json"
                if input_file.exists():
                    try:
                        df = pd.read_json(input_file, lines=True)
                        processed_data = []
                        for _, row in df.iterrows():
                            story = row.get(story_field, '')
                            processed_data.append({
                                'text': f"[LONG] {story}",
                                'type': 'long',
                                'source': ds_name
                            })
                        output_file = self.processed_dir / f"processed_{ds_name}_{split}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(processed_data, f, indent=2)
                    except Exception as e:
                        logger.error(f"Error processing {input_file}: {str(e)}")

def main():
    downloader = DatasetDownloader()
    
    # Download datasets
    downloader.download_writing_prompts()
    downloader.download_rocstories()
    downloader.download_hellaswag()  # Added hellaswag as an alternative
    downloader.download_gutenberg_books()  # Use default book IDs
    
    # Download alternative datasets
    downloader.download_all_alternatives()
    
    # Process datasets
    downloader.process_datasets()
    # Process alternative datasets
    downloader.process_alternative_datasets()

if __name__ == "__main__":
    main() 