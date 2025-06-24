from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

import argparse
import sys

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.2, 
                    model_name="gpt-4o", 
                    max_tokens=10000)

        self.example_prompt_template = PromptTemplate(
            input_variables=["user_question", "rewritten_question"],
            template="""
            <input query>
            {user_question}
            </input query>
            
            <rewritten question>
            {rewritten_question}
            </rewritten question>
            """
        )
        
        prefix = """
        <role>
        You are a clinical query-rewriting assistant specialized in Post-Acute Sequelae of COVID-19 (PASC, “Long-COVID”) patient education.
        You need to act as Clinicians, Doctor, Pharmaceutical assistant to transform user question to a more technical form and typo free.
        You need to understand that all of your user is a patient of PASC.
        </role>

        <tasks>
        1. **Rewrite the question** so that it:  
        • Names the key medical entity (symptom, treatment, test, etc.).  
        • Preserves any patient context given (age, gender, comorbidities, symptom duration).
        • Uses plain, direct language and adjust the query by sentence.

        2. **Do NOT fabricate missing clinical details.**  
        If critical context is absent, leave a placeholder in square brackets, e.g. “[age?]”.
        </tasks>
        
        Here are some examples:
        """
        
        suffix = """
        Now rewrite this question:
        <input query>
        {user_question}
        </input query>

        <output instructions>
        Only output the rewritten question after the transformation.
        Do NOT include the original question or any extra explanations.
        </output instructions>
        """
        
        self.examples = [
            {
                "user_question": "After I had COVID, I've been feeling really tired all the time, even after sleeping. Is that normal, and what's going on with my body?",
                "rewritten_question": "I have been experiencing persistent fatigue and exhaustion for [duration?] after recovering from COVID-19. What is the medical term for this, and what could be the underlying causes?"
            },
            {
                "user_question": "I've had trouble concentrating and remembering things since I got COVID. My brain feels foggy. Is that a long-term effect of the virus, and what can be done about it?",
                "rewritten_question": "Since recovering from COVID-19, I have noted significant cognitive impairments, including difficulty with concentration, memory recall, and overall mental clarity. Is this a recognized long-term complication of COVID-19 infection, and are there any treatments or management strategies available to address these cognitive difficulties?"
            },
            {
                "user_question": "I've noticed my sense of smell and taste have changed since I had COVID. Things just don't seem as flavorful or appealing. Is this something that will come back, or is this a permanent issue?",
                "rewritten_question": "Following my COVID-19 infection, I have experienced a significant alteration in my sense of smell (anosmia) and taste (ageusia). The quality and intensity of these senses have diminished considerably. What is the typical recovery timeline for olfactory and gustatory dysfunction after COVID-19, and are there any therapies or strategies to potentially improve these sensory functions?"
            }
        ]

        self.rewrite_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=["user_question"],
            example_separator="\n\n"
        )

        self.chain = (
            RunnablePassthrough()
            | self.rewrite_prompt
            | self.llm
        )

    def rewrite(self, user_question: str) -> str:
        response = self.chain.invoke({"user_question": user_question})
        return response.content.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rewrite medical queries about Long-COVID')
    parser.add_argument('query', nargs='?', help='Your medical question (enclose in quotes if contains spaces)')
    args = parser.parse_args()

    rewriter = QueryRewriter()
    
    result = rewriter.rewrite(args.query)
    print("\nRewritten Question:")
    print(result)