import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import logging
from src.models.story_generator import StoryPipeline
from src.models.rag import RAGGenerator
from src.models.outline_generator import OutlineGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate stories using AI models")
    
    # Basic arguments
    parser.add_argument("--prompt", type=str, required=True, help="Story prompt")
    parser.add_argument("--length", type=str, choices=["short", "long"], default="short", help="Story length")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--short-model", type=str, help="Path to short story model")
    parser.add_argument("--long-model", type=str, help="Path to long story model")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--max-length", type=int, help="Maximum story length")
    
    # Enhanced features
    parser.add_argument("--use-rag", action="store_true", help="Use RAG for enhanced generation")
    parser.add_argument("--use-outline", action="store_true", help="Generate outline for long stories")
    parser.add_argument("--style", type=str, choices=["creative", "formal", "casual"], default="creative", help="Writing style")
    
    return parser.parse_args()

def format_prompt(prompt: str, length: str, style: str) -> str:
    """Format the prompt based on length and style."""
    style_prompts = {
        "creative": "Write a creative and engaging",
        "formal": "Write a formal and structured",
        "casual": "Write a casual and conversational"
    }
    
    if length == "short":
        return f"{style_prompts[style]} short story in 2-3 paragraphs about: {prompt}"
    else:
        return f"{style_prompts[style]} detailed story with multiple chapters about: {prompt}"

def save_story(story: Dict, output_dir: str, filename: Optional[str] = None):
    """Save the generated story to a file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"story_{story['length']}_{hash(story['prompt'])}.json"
    
    with open(output_path / filename, 'w', encoding='utf-8') as f:
        json.dump(story, f, indent=2)

def main():
    args = parse_args()
    
    # Initialize pipeline
    pipeline = StoryPipeline(
        model_name=args.model_name,
        short_model_path=args.short_model,
        long_model_path=args.long_model
    )
    
    # Format prompt
    formatted_prompt = format_prompt(args.prompt, args.length, args.style)
    
    # Generate story
    if args.length == "long" and args.use_outline:
        # Use outline generator for long stories
        outline_gen = OutlineGenerator(pipeline)
        story = outline_gen.generate_with_outline(
            prompt=formatted_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length
        )
    else:
        # Use RAG if enabled
        if args.use_rag:
            rag_gen = RAGGenerator(pipeline)
            story = rag_gen.generate(
                prompt=formatted_prompt,
                length=args.length,
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=args.max_length
            )
        else:
            story = pipeline.generate(
                prompt=formatted_prompt,
                length=args.length,
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=args.max_length
            )
    
    # Save story
    save_story(story, args.output)
    logger.info(f"Story saved to {args.output}")

if __name__ == "__main__":
    main() 