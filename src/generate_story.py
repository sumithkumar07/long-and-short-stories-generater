import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import traceback
import os
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoryGenerator:
    def __init__(self, model_name="gpt2-medium"):
        try:
            logger.info(f"Loading pre-trained model: {model_name}")
            
            logger.info("Loading tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
            
            logger.info("Loading model...")
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Model loaded successfully")
            
            self.model.eval()
            logger.info("Model set to evaluation mode")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def generate_story(self, prompt, max_length=500, temperature=0.7, top_k=40, top_p=0.9, num_beams=5):
        try:
            logger.info(f"Generating story for prompt: {prompt}")
            
            # Format the prompt with clear story structure and context
            formatted_prompt = f"""Once upon a time, {prompt.lower()}

The story begins:

"""
            
            # Tokenize the prompt
            logger.info("Tokenizing prompt...")
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            
            # Generate story with improved parameters
            logger.info("Generating story...")
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_return_sequences=1,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    early_stopping=True
                )
            logger.info("Story generated successfully")
            
            # Decode and clean up the generated text
            logger.info("Decoding generated text...")
            story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output
            story = story.replace(formatted_prompt, "").strip()
            
            # Clean up any remaining formatting issues
            story = story.replace(" .", ".").replace(" ,", ",")
            story = "\n".join(line.strip() for line in story.split("\n") if line.strip())
            
            # Add proper story ending if missing
            if not any(story.endswith(end) for end in [".", "!", "?"]):
                story += "."
            
            logger.info("Story processing completed")
            return story
        except Exception as e:
            logger.error(f"Error generating story: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def save_story(prompt, story, filename):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated Story:\n{story}\n")
        f.write("-" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Story Generator Interactive Tool")
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode with default prompts')
    parser.add_argument('--max_length', type=int, default=800, help='Maximum story length (tokens)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Creativity (temperature)')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--num_beams', type=int, default=5, help='Beam search width')
    parser.add_argument('--output', type=str, default=None, help='File to save generated stories')
    args = parser.parse_args()

    logger.info("Initializing story generator...")
    story_generator = StoryGenerator()

    if args.interactive:
        print("\n=== Interactive Story Generation ===\n")
        output_file = args.output or f"generated_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        while True:
            prompt = input("Enter your story prompt (or type 'exit' to quit): ").strip()
            if prompt.lower() == 'exit':
                print(f"Stories saved to {output_file}")
                break
            story = story_generator.generate_story(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams
            )
            print(f"\nGenerated Story:\n{story}\n{'-'*80}")
            save_story(prompt, story, output_file)
    else:
        # Default batch mode
        test_prompts = [
            "A mysterious package arrives at the doorstep of a family home, containing an ancient artifact",
            "The old library held many secrets, including a hidden room with forbidden books",
            "A young inventor creates a time machine, but discovers an unexpected consequence",
            "The garden came alive at midnight, revealing magical creatures and hidden pathways",
            "A detective solves a case in a small town, uncovering a web of lies and deception"
        ]
        print("\n=== Generated Stories ===\n")
        output_file = args.output or f"generated_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            story = story_generator.generate_story(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams
            )
            print(f"Generated Story:\n{story}\n{'-'*80}")
            save_story(prompt, story, output_file)
        print(f"Stories saved to {output_file}")

if __name__ == "__main__":
    main() 