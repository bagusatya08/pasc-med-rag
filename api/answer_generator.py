import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

class AnswerGenerator:
    def __init__(self):
        """
        Initializes the OpenAI client to connect to the Hugging Face router.
        """
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")
        
        self.client = OpenAI(
            base_url="https://router.huggingface.co/novita/v3/openai",
            api_key=HF_TOKEN,
        )

    def generate(self, query: str, context: str) -> str:
        """
        Generates an answer using a model via the Hugging Face router.

        Args:
            query: The user's question.
            context: Retrieved context for answering the question.

        Returns:
            A single string containing the generated answer.
        """
        # We create a system message to provide context and a user message for the specific query.
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Use the following context to answer the user's question. Context: {context}"
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        try:
            # Make a non-streaming API call
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7, # You can adjust parameters as needed
                max_tokens=30000,
            )
            
            # Extract the text content from the response message
            return completion.choices[0].message.content.strip()

        except Exception as e:
            # Pass the error up to the main script to be handled
            print(f"An error occurred with the API call: {e}")
            raise