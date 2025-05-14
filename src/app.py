from flask import Flask, render_template, request, jsonify
import logging
import os
from generate_story import StoryGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize story generator with GPT-2 Medium
MODEL_NAME = "gpt2-medium"
story_generator = None

def initialize_generator():
    """Initialize the story generator."""
    global story_generator
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        story_generator = StoryGenerator(MODEL_NAME)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Initialize model when the app starts
initialize_generator()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate a story based on the prompt."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 800)
        temperature = data.get('temperature', 0.7)
        top_k = data.get('top_k', 40)
        top_p = data.get('top_p', 0.9)
        num_beams = data.get('num_beams', 5)
        
        if not prompt:
            return jsonify({'error': 'Please provide a prompt'}), 400
            
        story = story_generator.generate_story(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams
        )
        
        return jsonify({'story': story})
        
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 