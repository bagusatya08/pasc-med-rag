from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

class ContextSummarizer:
    def __init__(self, model_name="gpt-4o", temperature=0, max_tokens=3000):
        self.llm = ChatOpenAI(temperature=0, 
                              model_name="gpt-4o", 
                              max_tokens=3000)

        self.summarization_template = """
        <role>
        You are a clinical context summarization assistant in first person point of view. 
        Summarize the context retrieved chunks, while keeping the semantic and important medical information.
        </role>

        <input query>
        {context_retrieved}
        </input query>

        <tasks>
        1. **Summarize the query** so that it:  
        • Having less than 250 tokens
        • preserve the key medical entity (symptom, treatment, test, etc.).  
        • Preserves any patient context given (age, gender, comorbidities, symptom duration).   
        • Uses plain, direct language.

        <output (exactly)>
        Only output the final result of the context summarization (< 100 tokens).
        Do NOT include any extra explanations.
        </output (exactly)>
        """

        self.summarization_prompt = PromptTemplate(
            input_variables=["context_retrieved"],
            template=self.summarization_template
        )

        self.chain = (
            RunnablePassthrough.assign()
            | self.summarization_prompt
            | self.llm
        )

    def summarize_context(self, context_retrieved: str) -> str:
        response =self.chain.invoke({"context_retrieved": context_retrieved})
        return response.content.strip()