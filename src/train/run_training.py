import logging
from pathlib import Path
from download_model import download_and_save_model
from train_models import StoryTrainer
from config import TrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create necessary directories
    base_dir = Path("models/base")
    trained_dir = Path("models/trained")
    data_dir = Path("data")
    
    # Download and save base model if not already downloaded
    if not base_dir.exists():
        logger.info("Downloading base model...")
        download_and_save_model(save_dir=str(base_dir))
    else:
        logger.info("Base model already exists, skipping download")
    
    # Load configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = StoryTrainer(config)
    
    # Train models
    logger.info("Starting training process...")
    trainer.train(
        data_dir=str(data_dir),
        output_dir=str(trained_dir)
    )
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 