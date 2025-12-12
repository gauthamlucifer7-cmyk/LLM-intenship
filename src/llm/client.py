import openai
import os

class LLMClient:
    def __init__(self, model="gpt-4.1"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def chat(self, messages: list):
        return openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
      
