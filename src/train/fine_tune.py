import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoryFineTuner:
    def __init__(self, model_name="gpt2-medium", output_dir=None):
        self.model_name = model_name
        self.output_dir = output_dir or f"models/fine_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def prepare_data(self, train_file, eval_file=None, block_size=128):
        """Prepare the dataset for training."""
        logger.info("Preparing datasets...")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create training dataset
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=block_size
        )
        
        # Create evaluation dataset if provided
        eval_dataset = None
        if eval_file:
            eval_dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=eval_file,
                block_size=block_size
            )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        return train_dataset, eval_dataset, data_collator
    
    def train(self, train_dataset, eval_dataset=None, data_collator=None, 
              epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune the model on the provided dataset."""
        logger.info("Initializing model...")
        
        # Load model
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        model.to(self.device)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_steps=500,
            warmup_steps=500,
            learning_rate=learning_rate,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            save_total_limit=2,
            no_cuda=not torch.cuda.is_available(),
            local_rank=-1,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            optim="adamw_torch"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def prepare_story_data(input_file, output_file):
    """Convert story data into a format suitable for training."""
    logger.info(f"Preparing story data from {input_file}")
    
    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        
        # Format each story with clear start and end markers
        formatted_stories = []
        for story in stories:
            if isinstance(story, dict) and 'text' in story:
                formatted_story = f"<|startoftext|>\n{story['text'].strip()}\n<|endoftext|>\n"
                formatted_stories.append(formatted_story)
            elif isinstance(story, str):
                formatted_story = f"<|startoftext|>\n{story.strip()}\n<|endoftext|>\n"
                formatted_stories.append(formatted_story)
        
        # Write formatted stories to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(formatted_stories)
        
        logger.info(f"Formatted stories saved to {output_file}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Error reading JSON file {input_file}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing stories: {str(e)}")
        raise

def main():
    try:
        # Configuration
        model_name = "gpt2-medium"
        train_file = "data/processed/train_stories.txt"
        eval_file = "data/processed/eval_stories.txt"
        
        # Prepare data
        logger.info("Preparing training data...")
        prepare_story_data("data/processed_short_stories.json", train_file)
        
        logger.info("Preparing evaluation data...")
        prepare_story_data("data/processed_long_stories.json", eval_file)
        
        # Initialize fine-tuner
        fine_tuner = StoryFineTuner(model_name=model_name)
        
        # Prepare datasets
        train_dataset, eval_dataset, data_collator = fine_tuner.prepare_data(
            train_file=train_file,
            eval_file=eval_file,
            block_size=128
        )
        
        # Train the model
        trainer = fine_tuner.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            epochs=3,
            batch_size=4,
            learning_rate=5e-5
        )
        
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 