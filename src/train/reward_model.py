import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class StoryRewardModel(nn.Module):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        rewards = self.reward_head(last_hidden_state)
        return rewards.squeeze(-1)
    
    def compute_reward(self, text: str) -> float:
        """Compute reward for a given text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            reward = self(inputs["input_ids"], inputs["attention_mask"])
        return reward.item()

class HumanFeedbackCollector:
    def __init__(self, reward_model: StoryRewardModel):
        self.reward_model = reward_model
        self.feedback_buffer = []
        
    def collect_feedback(self, story: str, human_rating: float) -> None:
        """Collect human feedback for a story"""
        model_reward = self.reward_model.compute_reward(story)
        self.feedback_buffer.append({
            "story": story,
            "human_rating": human_rating,
            "model_reward": model_reward
        })
        
    def update_reward_model(self, learning_rate: float = 1e-5) -> None:
        """Update reward model based on collected feedback"""
        if not self.feedback_buffer:
            return
            
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for feedback in self.feedback_buffer:
            story = feedback["story"]
            target_reward = torch.tensor(feedback["human_rating"], dtype=torch.float)
            
            inputs = self.reward_model.tokenizer(story, return_tensors="pt", truncation=True, max_length=512)
            predicted_reward = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
            
            loss = criterion(predicted_reward, target_reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        self.feedback_buffer = []  # Clear buffer after update 