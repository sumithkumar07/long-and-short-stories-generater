# AI Story Generator

A web application that generates creative stories using GPT-2 Medium model. The application provides a user-friendly interface to generate stories of different lengths and creativity levels.

## Features

- Generate stories using GPT-2 Medium model
- Adjustable story length (Short, Medium, Long)
- Customizable creativity level
- Copy to clipboard functionality
- Modern, responsive UI with Tailwind CSS
- Loading indicators and error handling

## Project Structure

```
.
├── src/
│   ├── app.py              # Flask application
│   ├── generate_story.py   # Story generation logic
│   ├── templates/          # HTML templates
│   │   └── index.html     # Main UI template
│   └── train/             # Training scripts
├── data/                  # Training and evaluation data
├── models/               # Saved models
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sumithkumar07/long-and-short-stories-generater.git
cd long-and-short-stories-generater
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python src/app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter your story prompt and adjust the settings:
   - Story Length: Choose between Short (400 words), Medium (800 words), or Long (1200 words)
   - Creativity Level: Select from More Focused, Balanced, or More Creative

4. Click "Generate Story" to create your story

## Development

### Adding New Features

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

### Running Tests

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- GPT-2 model by OpenAI
- Flask web framework
- Tailwind CSS for styling 