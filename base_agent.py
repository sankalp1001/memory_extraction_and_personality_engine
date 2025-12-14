"""
Base Conversational Agent

A simple LLM-based agent that accepts a system prompt and generates responses.
The agent is personality-agnostic - personality is injected via the system prompt.
"""

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Default neutral prompt (no personality)
NEUTRAL_PROMPT = "You are a helpful assistant."


class BaseAgent:
    """
    Base conversational agent that generates responses using a configurable system prompt.
    
    The agent itself has no personality - personality is controlled by the system prompt
    provided by the PersonalityEngine.
    
    Usage:
        agent = BaseAgent()
        
        # Neutral response
        response = agent.respond("I had a stressful day")
        
        # With custom system prompt (from PersonalityEngine)
        response = agent.respond("I had a stressful day", system_prompt=personality_prompt)
    """

    def __init__(self, model: str = "openai/gpt-oss-20b", temperature: float = 0.7):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def respond(self, user_message: str, system_prompt: str = None) -> str:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's input message
            system_prompt: Optional system prompt (defaults to neutral)
        
        Returns:
            The agent's response string
        """
        if system_prompt is None:
            system_prompt = NEUTRAL_PROMPT
        
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content.strip()

