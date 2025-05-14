import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional
import wandb
from tqdm import tqdm
from config import TrainingConfig
from rag import StoryRetriever
from outline_generator import OutlineGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoryDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = self.encodings["input_ids"].clone()
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
    
    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

class StoryTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._initialize_model()
        self.retriever = StoryRetriever() if config.rag_config["enabled"] else None
        self.outline_generator = OutlineGenerator(config.base_model_path)
        
    def _initialize_model(self):
        """Initialize model and tokenizer from local path"""
        logger.info(f"Loading model from {self.config.base_model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.base_model_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.config.base_model_path)
        self.model.to(self.device)
        
    def load_data(self, data_dir: str) -> Dict[str, List[str]]:
        """Load and prepare training data"""
        data_dir = Path(data_dir)
        stories = {
            "short": [],
            "long": []
        }
        
        # Load short stories
        short_stories_path = data_dir / "processed_short_stories.json"
        if short_stories_path.exists():
            with open(short_stories_path, "r", encoding="utf-8") as f:
                stories["short"] = [item["text"] for item in json.load(f)]
                
        # Load long stories
        long_stories_path = data_dir / "processed_long_stories.json"
        if long_stories_path.exists():
            with open(long_stories_path, "r", encoding="utf-8") as f:
                stories["long"] = [item["text"] for item in json.load(f)]
                
        return stories
    
    def train(self, data_dir: str, output_dir: str):
        """Train both short and long story models"""
        if self.config.use_wandb:
            wandb.init(project="story-generator")
            
        stories = self.load_data(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RAG if enabled
        if self.retriever:
            self.retriever.load_stories(data_dir)
        
        # Train short story model
        if stories["short"]:
            logger.info("Training short story model...")
            short_model_dir = output_dir / "short_story_model"
            self._train_model(
                stories["short"],
                short_model_dir,
                "short_story",
                self.config.short_story_config
            )
            
        # Train long story model
        if stories["long"]:
            logger.info("Training long story model...")
            long_model_dir = output_dir / "long_story_model"
            self._train_model(
                stories["long"],
                long_model_dir,
                "long_story",
                self.config.long_story_config
            )
            
        if self.config.use_wandb:
            wandb.finish()
            
    def _train_model(self, texts: List[str], output_dir: Path, model_type: str, config: Dict):
        """Train a single model"""
        # Create dataset and dataloader
        dataset = StoryDataset(texts, self.tokenizer, config["max_length"])
        
        # Split into train and validation sets (80-20 split)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            save_total_limit=2,
            report_to="wandb" if self.config.use_wandb else "none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Model saved to {output_dir}")
        
        # Log final metrics
        if self.config.use_wandb:
            final_metrics = trainer.evaluate()
            wandb.log(final_metrics)

if __name__ == "__main__":
    # Example usage
    config = TrainingConfig()
    trainer = StoryTrainer(config)
    
    trainer.train(
        data_dir="data/processed",
        output_dir="models/trained"
    ) 