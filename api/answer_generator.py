import os
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from typing import List, Dict
from langchain_openai import ChatOpenAI

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
        
        self.examples = [
            {
                "query": "What are common symptoms of influenza?",
                "answer": "Influenza typically presents with fever, chills, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. Some patients may experience vomiting and diarrhea, though this is more common in children."
            },
            {
                "query": "How is type 2 diabetes diagnosed?",
                "answer": "Type 2 diabetes is diagnosed through blood tests. Common tests include the fasting plasma glucose test (FPG), the A1C test, and the oral glucose tolerance test (OGTT). Diagnosis typically requires two abnormal test results on different days."
            },
            {
                "query": "What is the recommended treatment for malaria?",
                "answer": "I don't have enough information to answer that question. Malaria treatment depends on the specific parasite species, patient age, pregnancy status, and severity of illness. Please consult a tropical disease specialist for appropriate treatment recommendations."
            }
        ]
        
        self.system_template = (
            "<role>\n"
            "You are a medical assistant. Use the following context to answer the user's question.\n"
            "</role>\n\n"
            
            "<context>\n"
            "{context}\n"
            "</context>\n\n"
            
            "<instructions>\n"
            "1. Answer based ONLY on the provided context\n"
            "2. Maintain a professional medical tone\n"
            "3. If context doesn't contain relevant information, say 'I don't have enough information'\n"
            "4. Follow the response style shown in the examples\n"
            "</instructions>"
        )
        self.system_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{query}"),
            ("ai", "{answer}")
        ])
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples,
        )
        
        self.final_prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            self.few_shot_prompt,
            HumanMessagePromptTemplate.from_template("{query}")
        ])

    def generate(self, query: str, context: str) -> str:
        """Generate answer using context with fixed few-shot examples"""
        formatted_prompt = self.final_prompt.format_messages(
            context=context,
            query=query
        )
        
        response = self.llm.invoke(formatted_prompt)
        return response.content.strip()