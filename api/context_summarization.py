from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

import argparse
import sys

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

class ContextSummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.4, 
                              model_name="gpt-4o", 
                              max_tokens=10000)
        
        self.example_prompt_template = PromptTemplate(
            input_variables=["contexts", "summarized_contexts"],
            template="""
            <input contexts>
            {contexts}
            </input contexts>
            
            <summarized contexts>
            {summarized_contexts}
            </summarized contexts>
            """
        )

        prefix= """
        <role>
        You are a clinical context summarization assistant in first person point of view. 
        Summarize the context retrieved chunks, while keeping the semantic and important medical information.
        </role>

        <tasks>
        1. **Summarize the query** so that it:  
        • Maintained Token Limit: Ensured each summary stays within the specified token limit.
        • preserve the key medical entity (symptom, treatment, test, etc.).  
        • Preserves any patient context given (age, gender, comorbidities, symptom duration).   
        • Focused on Key Information: Emphasized the essential details about the prevalence of fatigue in Long COVID.
        • Improved Clarity: Reworded sentences for better flow and easier understanding.

        2. **Clean the contexts from noise**:
        • Deleted References: Removed the citations to focus on the core content.
        • Deleted Numbers: Removed the numbers from the citation, as the context is already established.

        Here are some examples:
        """

        suffix="""
        Now summarize the contexts given:
        <input contexts>
        {contexts}
        </input contexts>

        <output (exactly)>
        Only output the final result of the context summarization.
        Do NOT include any extra explanations.
        </output (exactly)>
        """

        self.examples = [
            {
                "contexts": """
                <Chunk 1>
                COVID and that symptomatology is not dominated by single or-
                gan dysfunction. Fatigue was the most consistent symptom across
                both Timepoints 1 and 2 and has been reported elsewhere as a
                very common symptom of long COVID; 79% of individuals at 23
                weeks reported severe fatigue in 1 cohort of a long COVID support
                group, albeit this was improved from 87% at 10 weeks ( Van Herck
                et al., 2021 ). Underlying mechanisms of persistent immune activa- 
                
                <Chunk 2>
                COVID-19 syndrome, Long COVID, Long-Haul COVID-19 Fatigue, Internet therapy, Study protocol, Randomised
                controlled trial
                Background
                The coronavirus disease 2019 (COVID-19) pandemic is
                a serious health crisis resulting in multiple symptoms in
                a substantial subgroup of patients. During acute
                COVID-19, up to 4 weeks after the onset of the infection
                [1], severe fatigue is one of the most prevalent symptoms
                [2– 5]. When symptoms of COVID-19 continue for more
                than 12 weeks and are not explained by an alternative

                <Chunk 3>
                COVID at approximately 3 and 6 months after the infection. Furthermore, our longitudinal follow-up data suggest that fatigue
                does not resolve over time in all patients, even if they receive
                health care. In addition, patients experience both physical and
                mental fatigue. Fatigue is the most prominent symptom in patients with long
                COVID [2, 4], irrespective of the severity of the initial infection
                [14]. Nevertheless, most studies are cross-sectional and use a
                """,
                "summarized_contexts": """
                <Chunk 1>
                COVID symptomatology is not organ-specific. Fatigue, the most consistent symptom across Timepoints 1 and 2, is common in Long COVID. 
                79% of Long COVID support group participants reported severe fatigue at 23 weeks, improving from 87% at 10 weeks. 
                Underlying immune activation is hypothesized.

                <Chunk 2>
                The COVID-19 pandemic causes multiple symptoms in a subgroup of patients. 
                Severe fatigue is a common symptom of acute COVID-19, 
                persisting for more than 12 weeks in a significant proportion of individuals, and is not explained by other conditions.

                <Chunk 3>
                Fatigue persists in some Long COVID patients, even with healthcare. 
                Physical and mental fatigue are also common. Fatigue is the most prominent symptom in Long COVID, 
                regardless of initial infection severity, although most studies are cross-sectional.
                """
            }
        ]

        self.summarization_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=["contexts"],
            example_separator="\n\n"
        )

        self.chain = (
            RunnablePassthrough.assign()
            | self.summarization_prompt
            | self.llm
        )

    def summarize(self, contexts: str) -> str:
        response =self.chain.invoke({"contexts": contexts})
        return response.content.strip()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarized medical retrieval result of Long Covid Article Chunks')
    parser.add_argument('query', nargs='?', help='Retrieval Result (enclose in quotes if contains spaces)')
    args = parser.parse_args()

    summary = ContextSummarizer()
    
    result = summary.summarize(args.query)
    print("\nSummarized Contexts:")
    print(result)