import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

class StoryRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.story_embeddings = None
        self.stories = []
        self.index = None
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using the sentence transformer model"""
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def load_stories(self, data_dir: str):
        """Load stories and compute their embeddings"""
        data_dir = Path(data_dir)
        
        # Load short stories
        short_stories_path = data_dir / "processed_short_stories.json"
        if short_stories_path.exists():
            with open(short_stories_path, "r", encoding="utf-8") as f:
                short_stories = json.load(f)
                for story in short_stories:
                    self.stories.append({
                        "text": story["text"],
                        "type": "short",
                        "prompt": story.get("prompt", ""),
                        "genre": story.get("genre", "general"),
                        "themes": story.get("themes", []),
                        "characters": story.get("characters", [])
                    })
        
        # Load long stories
        long_stories_path = data_dir / "processed_long_stories.json"
        if long_stories_path.exists():
            with open(long_stories_path, "r", encoding="utf-8") as f:
                long_stories = json.load(f)
                for story in long_stories:
                    self.stories.append({
                        "text": story["text"],
                        "type": "long",
                        "prompt": story.get("prompt", ""),
                        "genre": story.get("genre", "general"),
                        "themes": story.get("themes", []),
                        "characters": story.get("characters", [])
                    })
        
        # Compute embeddings
        logger.info("Computing story embeddings...")
        embeddings = []
        for story in self.stories:
            # Combine story text with metadata for better retrieval
            story_context = f"{story['text']} {story['genre']} {' '.join(story['themes'])} {' '.join(story['characters'])}"
            embedding = self._get_embedding(story_context)
            embeddings.append(embedding)
        
        self.story_embeddings = np.vstack(embeddings)
        
        # Create FAISS index for efficient similarity search
        dimension = self.story_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.story_embeddings.astype('float32'))
        
        logger.info(f"Loaded {len(self.stories)} stories with embeddings")
    
    def retrieve_relevant_stories(
        self,
        prompt: str,
        story_type: str,
        top_k: int = 3,
        genre: str = None,
        themes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant stories based on prompt similarity and metadata"""
        # Get prompt embedding
        prompt_embedding = self._get_embedding(prompt)
        
        # Search in FAISS index
        distances, indices = self.index.search(
            prompt_embedding.reshape(1, -1).astype('float32'),
            k=len(self.stories)
        )
        
        # Filter and rank results
        filtered_stories = []
        for idx in indices[0]:
            story = self.stories[idx]
            
            # Apply filters
            if story["type"] != story_type:
                continue
            if genre and story["genre"] != genre:
                continue
            if themes and not any(theme in story["themes"] for theme in themes):
                continue
                
            filtered_stories.append({
                "story": story,
                "similarity": 1 / (1 + distances[0][idx])  # Convert distance to similarity
            })
            
            if len(filtered_stories) >= top_k:
                break
        
        return [item["story"] for item in filtered_stories]
    
    def get_relevant_context(
        self,
        prompt: str,
        story_type: str,
        top_k: int = 3,
        genre: str = None,
        themes: List[str] = None
    ) -> str:
        """Get relevant context from retrieved stories"""
        relevant_stories = self.retrieve_relevant_stories(
            prompt, story_type, top_k, genre, themes
        )
        
        context_parts = []
        for story in relevant_stories:
            # Extract relevant elements based on story type
            if story_type == "short":
                context_parts.append(f"Short story example:\n{story['text']}\n")
            else:
                # For long stories, include structure and themes
                context_parts.append(
                    f"Long story example:\n"
                    f"Genre: {story['genre']}\n"
                    f"Themes: {', '.join(story['themes'])}\n"
                    f"Characters: {', '.join(story['characters'])}\n"
                    f"Excerpt: {story['text'][:500]}...\n"
                )
        
        return "\n".join(context_parts)
    
    def get_writing_guidelines(self, story_type: str, genre: str = None) -> str:
        """Get writing guidelines based on story type and genre"""
        guidelines = []
        
        if story_type == "short":
            guidelines.extend([
                "Focus on a single theme or message",
                "Keep the plot concise and focused",
                "Develop 1-2 main characters",
                "Use vivid imagery and sensory details",
                "End with a strong conclusion"
            ])
        else:
            guidelines.extend([
                "Develop a complex plot with multiple subplots",
                "Create well-rounded characters with arcs",
                "Build tension and conflict throughout",
                "Include detailed world-building",
                "Maintain consistent pacing"
            ])
            
        if genre:
            genre_guidelines = {
                "fantasy": [
                    "Include magical elements and world-building",
                    "Create unique creatures or races",
                    "Develop a magic system"
                ],
                "mystery": [
                    "Plant clues throughout the story",
                    "Create red herrings",
                    "Build suspense and tension"
                ],
                "romance": [
                    "Develop chemistry between characters",
                    "Include emotional conflict",
                    "Build romantic tension"
                ]
            }
            guidelines.extend(genre_guidelines.get(genre, []))
            
        return "\n".join(f"- {guideline}" for guideline in guidelines) 