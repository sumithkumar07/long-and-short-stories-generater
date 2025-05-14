import pytest
from src.generate_story import StoryGenerator

def test_story_generator_initialization():
    """Test if the story generator initializes correctly."""
    generator = StoryGenerator()
    assert generator is not None
    assert generator.model is not None
    assert generator.tokenizer is not None

def test_story_generation():
    """Test if story generation works with basic parameters."""
    generator = StoryGenerator()
    prompt = "A boy found a magic pencil"
    story = generator.generate_story(
        prompt=prompt,
        max_length=100,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        num_beams=5
    )
    assert isinstance(story, str)
    assert len(story) > 0

def test_story_generation_parameters():
    """Test story generation with different parameters."""
    generator = StoryGenerator()
    prompt = "A mysterious door appeared in the wall"
    
    # Test with different lengths
    short_story = generator.generate_story(prompt, max_length=200)
    long_story = generator.generate_story(prompt, max_length=800)
    assert len(short_story) < len(long_story)
    
    # Test with different temperatures
    focused_story = generator.generate_story(prompt, temperature=0.5)
    creative_story = generator.generate_story(prompt, temperature=0.9)
    assert focused_story != creative_story

def test_error_handling():
    """Test error handling for invalid inputs."""
    generator = StoryGenerator()
    
    # Test empty prompt
    with pytest.raises(Exception):
        generator.generate_story("")
    
    # Test invalid max_length
    with pytest.raises(Exception):
        generator.generate_story("test", max_length=-1)
    
    # Test invalid temperature
    with pytest.raises(Exception):
        generator.generate_story("test", temperature=2.0) 