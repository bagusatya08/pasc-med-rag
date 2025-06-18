from langchain_openai import ChatOpenAI
# from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

class QueryRewriter:
    def __init__(self, model_name="gpt-4o", temperature=0, max_tokens=1500):
        self.llm = ChatOpenAI(temperature=0, 
                    model_name="gpt-4o", 
                    max_tokens=1500)

        self.rewrite_template = """
        <role>
        You are a clinical query-rewriting assistant. Transform the user’s informal health question about Post-Acute Sequelae of COVID-19 (PASC, “Long-COVID”) into a structured, RAG-ready query.
        </role>

        <input query>
        {original_query}
        </input query>

        <tasks>
        1. **Rewrite the query** so that it:  
        • Names the key medical entity (symptom, treatment, test, etc.).  
        • Preserves any patient context given (age, gender, comorbidities, symptom duration).
        • Uses plain, direct language (≤ 25 words) and one main question.

        2. **Do NOT fabricate missing clinical details.**  
        If critical context is absent, leave a placeholder in square brackets, e.g. “[age?]”.
        </tasks>

        <output (exactly)>
        Only output the single-sentence rewritten query.
        Do NOT include the original query or any extra explanations.
        </output (exactly)>
        """

        self.examples = [
            {
                "original_query": "",
                "Final Query":""
            },
            {
                "original_query": "",
                "Final Query":""
            },
            {
                "original_query": "",
                "Final Query":""
            }
            ]

        self.rewrite_prompt = PromptTemplate(
            examples=self.examples,
            input_variables=["original_query"],
            template=self.rewrite_template
        )

        self.chain = (
            RunnablePassthrough.assign()
            | self.rewrite_prompt
            | self.llm
        )

    def rewrite(self, original_query: str) -> str:
        response = self.chain.invoke({"original_query": original_query})
        return response.content.strip()