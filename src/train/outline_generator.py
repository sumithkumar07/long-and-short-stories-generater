from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict
import logging
from rag import StoryRetriever

logger = logging.getLogger(__name__)

class OutlineGenerator:
    def __init__(self, model_path: str, retriever: StoryRetriever = None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.retriever = retriever
        
    def generate_outline(
        self,
        prompt: str,
        num_chapters: int = 5,
        genre: str = None,
        themes: List[str] = None
    ) -> List[Dict[str, str]]:
        """Generate a chapter-by-chapter outline for a long story"""
        # Get relevant context and guidelines
        context = ""
        if self.retriever:
            context = self.retriever.get_relevant_context(
                prompt, "long", top_k=3, genre=genre, themes=themes
            )
            guidelines = self.retriever.get_writing_guidelines("long", genre)
        else:
            guidelines = ""
            
        outline_prompt = f"""Context: {context}

Writing Guidelines:
{guidelines}

Create a detailed outline for a story about: {prompt}
The story should have {num_chapters} chapters. For each chapter, provide:
1. Chapter title
2. Main events
3. Character development
4. Key plot points
5. Themes and motifs to explore

Outline:"""
        
        # Generate outline
        inputs = self.tokenizer(outline_prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        outline_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse outline into chapters
        chapters = self._parse_outline(outline_text)
        return chapters
    
    def _parse_outline(self, outline_text: str) -> List[Dict[str, str]]:
        """Parse the generated outline text into structured chapter information"""
        chapters = []
        current_chapter = {}
        
        lines = outline_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Chapter"):
                if current_chapter:
                    chapters.append(current_chapter)
                current_chapter = {
                    "title": line,
                    "events": [],
                    "character_development": [],
                    "plot_points": [],
                    "themes": []
                }
            elif "Main events:" in line:
                current_section = "events"
            elif "Character development:" in line:
                current_section = "character_development"
            elif "Key plot points:" in line:
                current_section = "plot_points"
            elif "Themes and motifs:" in line:
                current_section = "themes"
            elif current_chapter and line.startswith("-"):
                current_chapter[current_section].append(line[1:].strip())
        
        if current_chapter:
            chapters.append(current_chapter)
            
        return chapters
    
    def expand_chapter(
        self,
        chapter: Dict[str, str],
        context: str = "",
        genre: str = None,
        themes: List[str] = None
    ) -> str:
        """Expand a chapter outline into a full chapter"""
        # Get additional context and guidelines
        if self.retriever:
            story_context = self.retriever.get_relevant_context(
                chapter["title"], "long", top_k=2, genre=genre, themes=themes
            )
            guidelines = self.retriever.get_writing_guidelines("long", genre)
            context = f"{context}\n\nRelevant examples:\n{story_context}\n\nWriting guidelines:\n{guidelines}"
        
        chapter_prompt = f"""Context: {context}

Chapter Title: {chapter['title']}

Main Events:
{chr(10).join('- ' + event for event in chapter['events'])}

Character Development:
{chr(10).join('- ' + dev for dev in chapter['character_development'])}

Key Plot Points:
{chr(10).join('- ' + point for point in chapter['plot_points'])}

Themes to Explore:
{chr(10).join('- ' + theme for theme in chapter['themes'])}

Write a detailed chapter based on this outline. Focus on:
1. Vivid descriptions and sensory details
2. Character emotions and motivations
3. Pacing and tension
4. Theme development
5. Foreshadowing future events

Chapter:"""
        
        inputs = self.tokenizer(chapter_prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 