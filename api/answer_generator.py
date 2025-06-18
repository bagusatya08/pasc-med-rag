import os
from dotenv import load_dotenv
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import List, Dict, Optional

from langchain_community.chat_models import ChatOpenAI

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

class AnswerGenerator:
    def __init__(self, model_name=MODEL_NAME, temperature=0.7, max_tokens=30000):
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://router.huggingface.co/novita/v3/openai",
            openai_api_key=HF_TOKEN,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.qa_template = """
        <role>
        You are a medical assistant. Use the following context to answer the user's question.
        </role>
        
        <context>
        {context}
        </context>
        
        <instructions>
        1. Answer the question based ONLY on the provided context
        2. Maintain a professional medical tone
        3. If the context doesn't contain relevant information, say "I don't have enough information"
        4. Follow the response style shown in the examples
        </instructions>
        """
        
        self.example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{query}"),
            ("ai", "{answer}")
        ])
        
        self.qa_chain = None

    def generate(
        self,
        query: str,
        context: str,
        examples: Optional[List[Dict]] = None
    ) -> str:
        """Generate answer using context and optional few-shot examples"""
        if examples:
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=self.example_prompt,
                examples=examples,
            )
            messages = [
                ("system", self.qa_template),
                few_shot_prompt,
                ("human", "{query}")
            ]
        else:
            messages = [
                ("system", self.qa_template),
                ("human", "{query}")
            ]

        qa_prompt = ChatPromptTemplate.from_messages(messages)
        
        self.qa_chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | qa_prompt
            | self.llm
        )