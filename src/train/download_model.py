import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_save_model(model_name: str = "gpt2", save_dir: str = "models/base"):
    """
    Download and save the model locally.
    
    Args:
        model_name: Name of the model to download
        save_dir: Directory to save the model
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_name} model...")
    
    # Download and save tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(save_path)
    
    # Download and save model
    config = GPT2Config.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    model.save_pretrained(save_path)
    
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    download_and_save_model() 