from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import  ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

import os
import regex as re

import argparse
import sys

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

class AnswerGeneratorRAG:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_base="https://router.huggingface.co/novita/v3/openai",
            openai_api_key=HF_TOKEN,
            temperature=0.1,
            max_tokens=30000
        )
        
        self.example_prompt_template = PromptTemplate(
            input_variables=["user_question", "contexts", "reasoning", "answer"],
            template="""
            <input question>
            {user_question}
            </input question>
            
            <context>
            {context}
            </context>
            
            <reasoning>
            {reasoning}
            </reasoning>
            
            <answer>
            {answer}
            </answer>
            """
        )
        
        prefix = """
        <role>  
        You are a medical expert in Long COVID (PASC), speaking as a supportive doctor to a diagnosed patient. 
        Use only the provided context to give accurate, evidence-based answers.  
        </role>  

        <reasoning>  
        Before answering:  
        1. **Understand the question** – What is the patient asking?  
        2. **Analyze context** – Which chunks are relevant? What evidence supports the answer? Any contradictions?  
        3. **Plan response** – How can this be explained simply and clearly to a patient?  
        </reasoning>  

        <instructions>  
        When answering:  
        1. Use only the context; if missing, say: “I don’t have enough information.”  
        2. Use plain, patient-friendly language.  
        3. Be warm and empathetic.  
        4. Structure your response in two parts:  
        - **Part 1: Empathetic acknowledgment** (validate the question)  
        - **Part 2: Clear explanation** (simple, medically accurate answer)  
        </instructions>
        """
        suffix = """
        Now generate an answer for this question using the provided context:
        <input question>
        {user_question}
        </input question>
        
        <context>
        {context}
        </context>
        """
        
        self.examples = [
            {
                "user_question": "Can someone vaccinated get PASC infection?",
                "context": """
                <Chunk 1>
                the belief that
                vaccination was
                contraindicated
                because of PASC
                (15. 6%). Schulthei ß
                et al., 2021
                Germany Cross-sectional DigiHero cohort
                (recruited via
                direct mailing to
                citizens of Halle, Germany)
                n = 294
                Age range: > 14
                Median (IQR)
                age*: 51. 2
                (15 – 83)
                Median (IQR)
                • n = 137 (46. 6%)
                received 1 dose of a
                SARS-CoV-2 vaccine
                (mRNA or adenoviral
                • vector, or
                combination of both)
                • PASC defined as
                symptoms persisting
                more than 4 weeks from
                COVID-19 positive test
                
                <Chunk 2>
                persisted more than 12
                weeks
                ongoing symptoms, whereas 24 (38. 7%)
                unvaccinated
                individuals reported
                ongoing symptoms. Furthermore, the
                percentage of post-
                infection vaccinations
                was identical in
                patients with PASC
                that experienced
                resolution of their
                symptoms and in those
                that reported ongoing
                PASC. From the 175
                individuals with
                reported PASC, 104
                individuals still had
                ongoing symptoms at
                the time of analysis, while 71 had resolved
                PASC. Out of the 104
                individuals with
                
                <Chunk 3>
                PASC. Out of the 104
                individuals with
                ongoing symptoms, 80
                (76. 9%) were
                vaccinated post-
                infection, whereas out
                of the 71 individuals
                with resolved PASC, 54
                (76. 1%) were
                vaccinated post-
                infection. Simon et al., 2021 **
                United
                States
                Retrospective
                Cohort
                Arcadia Data
                Research
                n = 240 648
                Age range: N/A
                Mean (SD) age: N/A
                Sex (%F/%M): 59. 9/40. 1
                • n = 17 796 (7. 4%)
                individuals were
                vaccinated with one
                SARS-CoV-2 vaccine
                dose within the first
                twelve weeks after
                """,
                "reasoning": """
                1. Understand the question: The user is asking if people who are vaccinated against COVID-19 can still develop PASC.
                2. Context relevance:
                    - Chunk 1: Confirms that vaccinated individuals were part of the study population and that PASC was still observed in this group.
                    - Chunk 2: Indicates that vaccination did not prevent ongoing PASC symptoms, as both vaccinated and unvaccinated individuals reported them at similar rates.
                    - Chunk 3: Shows that a high proportion of people with ongoing PASC symptoms had received post-infection vaccination (76.9%), similar to those who recovered (76.1%), suggesting that vaccination did not completely prevent or resolve PASC.
                3. Response structuring:
                    - Start with an empathetic acknowledgment that it's a valid and common concern.
                    - Explain clearly that while vaccination is beneficial, it does not fully prevent PASC in all cases.
                """,
                "answer": """
                That’s a great question, and it’s completely understandable to wonder about the role of vaccination when it comes to Long COVID. 
                It’s important to know that you're not alone in asking this—many people are trying to make sense of how vaccines impact recovery and risk.
                
                While getting vaccinated is one of the best ways to reduce your risk of severe COVID-19, it does not fully eliminate the chance of developing PASC (Post-Acute Sequelae of COVID-19). 
                According to the data, some individuals who were vaccinated still went on to experience lingering symptoms like fatigue or brain fog. 
                In fact, studies show that similar percentages of vaccinated individuals reported ongoing PASC symptoms as those who did not. 
                So yes, even if you're vaccinated, it is still possible—though perhaps less likely—to develop PASC.
                """
            },
            {
                "user_question": "What is the distinct difference in normal COVID-19 infection and PASC infection?",
                "context": """
                <Chunk 1>
                sistent fatigue with bodily pain or psychological prob -
                lems, neurocognitive problems, or ongoing respiratory
                issues. Prevalence rates of PASC vary between different
                studies and subgroups, with the highest rates among
                women and those who required hospitalization for the
                initial SARS-CoV-2 infection [3–5]. Worldwide, physi -
                cians face considerable difficulties in treating PASC due
                to the absence of specific biological markers for clear
                diagnostic classification and the lack of causal thera -
                
                <Chunk 2>
                pulmonary recovery. Since PASC are found in as many as 10% of COVID- 19 patients (Sahanic et al., 2021; Venkatesan, 2021; Sudre et al., 2021b)⁠, robust, resource- saving tools assessing the individual risk of pulmo-
                nary complications are urgently needed (Shah et al., 2021; Raghu and Wilson, 2020)⁠. Covariates
                and characteristics of severe acute COVID- 19 such as male sex, age, and preexisting comorbidities,
                
                <Chunk 3>
                highly likely that this pathophysiology also occurs in patients with
                post- COVID- 19 syndrome; indeed, preliminary data from our first
                108 patients with post- COVID- 19 syndrome does demonstrate
                an increased incidence of elevation of IL- 6. 24 More recently, data
                have emerged demonstrating decreases in CD8+ T cells in patients
                with PASC compared with healthy controls and patients who had
                COVID- 19 but did not develop PASC. 25 26 Patients with PASC have
                Patient’s perspective Continued
                """,
                "reasoning": """
                1. Understand the question: Differences of COVID-19 and Long COVID-19
                2. Context analysis:
                   - Chunk 1: Highlights prevalence variation of PASC and diagnostic challenges due to lack of biomarkers
                   - Chunk 2: Emphasizes need for risk assessment tools and identifies risk factors (sex, age, comorbidities)
                   - Chunk 3: Presents biological evidence (elevated IL-6, reduced CD8+ T cells) linked to PASC pathophysiology
                4. Response structuring:
                    - Start with an empathetic acknowledgment that it's a valid and common concern.
                    - Use simple comparison: active infection vs. long-term aftermath.
                """,
                "answer": """
                That’s an excellent and very important question. 
                Understanding the difference helps make sense of what you’re experiencing and how we can support your recovery.

                Simply put, a normal COVID-19 infection is the early, active phase—when the virus is in your body, causing symptoms like fever or cough, and you are contagious. 
                This usually resolves within a few weeks. PASC, or Long COVID, happens after that. It’s not a new infection, and you’re no longer contagious. 
                But symptoms like fatigue, “brain fog,” or breathing issues may linger for weeks or months. 
                It reflects ongoing effects of the virus on your body, especially the immune and nervous systems.
                """
            }
        ]

        self.generator_prompt = FewShotPromptTemplate(
                examples=self.examples,
                example_prompt=self.example_prompt_template,
                prefix=prefix,
                suffix=suffix,
                input_variables=["user_question", "context"],
                example_separator="\n\n"
            )
        
        self.chain = (
            RunnablePassthrough()
            | self.generator_prompt
            | self.llm
        )

    def generate(self, user_question: str, context: str) -> str:
        full_response = self.chain.invoke({
            "user_question": user_question,
            "context": context
        }).content.strip()
        
        answer_match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return full_response

class AnswerGeneratorFrozen:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_base="https://router.huggingface.co/novita/v3/openai",
            openai_api_key=HF_TOKEN,
            temperature=0.4,
            max_tokens=30000
        )
        
        self.example_prompt_template = PromptTemplate(
            input_variables=["user_question", "reasoning", "answer"],
            template="""
            <input question>
            {user_question}
            </input question>
            
            <reasoning>
            {reasoning}
            </reasoning>
            
            <answer>
            {answer}
            </answer>
            """
        )
        
        prefix = """
        <role>
        You are a medical expert in Long COVID (PASC), speaking as a supportive doctor to a diagnosed patient. 
        Use only the provided context to give accurate, evidence-based answers.
        </role>

        <reasoning>  
        Before answering:  
        1. **Understand the question** – What is the patient asking? 
        2. **Plan response** – How can this be explained simply and clearly to a patient?  
        </reasoning>  

        <instructions>  
        When answering:
        1. Use plain, patient-friendly language.  
        2. Be warm and empathetic.  
        3. Structure your response in two parts:  
        - **Part 1: Empathetic acknowledgment** (validate the question)  
        - **Part 2: Clear explanation** (simple, medically accurate answer)  
        </instructions>
        """
        
        suffix = """
        Now generate an answer for this question:
        <input question>
        {user_question}
        </input question>
        """
        
        self.examples = [
            {
                "user_question": "Can someone vaccinated get PASC infection?",
                "reasoning": """
                1. Understand the question: The user is asking if people who are vaccinated against COVID-19 can still develop PASC.
                2. Response structuring:
                    - Answer with "yes" or "no" answer followed with brief explanation
                    - Explain clearly that while vaccination is beneficial, it does not fully prevent PASC in all cases.
                """,
                "answer": """
                Yes, it is possible, though less likely, for vaccinated individuals to develop PASC.

                While COVID-19 vaccines are highly effective at preventing severe illness, hospitalization, and death from the virus, they do not offer 100% protection against infection itself. 
                If a vaccinated person experiences a "breakthrough infection"—meaning they still get infected with COVID-19 despite being vaccinated—there is still a chance, albeit a reduced one, that they could develop PASC afterward. 
                The good news is that studies consistently show that vaccination significantly lowers your risk of developing PASC compared to unvaccinated individuals who get COVID-19. So, while It is not a complete guarantee against PASC, vaccination remains one of the best tools we have to protect against both severe acute COVID-19 and its long-term complications.
                """
            },
            {
                "user_question": "What is the distinct difference in normal COVID-19 infection and PASC infection?",
                "reasoning": """
                1. Understanding the question: Differences of COVID-19 and Long COVID
                2. Response structuring:
                    - Start with an empathetic acknowledgment that it's a valid and common concern.
                    - Explain that the differences located in the type of conditions of COVID-19 and Long COVID
                """,
                "answer": """
                It is an excellent and very important question to distinguish between a normal COVID-19 infection and PASC. 
                Understanding this difference is crucial for us to manage your health concerns effectively.

                Simply put, a normal COVID-19 infection refers to the acute illness when the SARS-CoV-2 virus is actively in your body, causing symptoms like fever, cough, or loss of smell, and you are contagious. 
                This phase typically lasts a few weeks. PASC (Post-Acute Sequelae of SARS-CoV-2 infection), or Long COVID, is what happens after that initial active infection has largely resolved. 
                It is not a new infection, and you are no longer contagious. Instead, PASC involves persistent or new health problems—such as profound fatigue, "brain fog," or shortness of breath—that linger for weeks or months following your acute COVID-19 illness. 
                So, while the initial infection is the direct viral attack, PASC represents the ongoing, complex aftermath.
                """

            }
        ]

        self.generator_prompt = FewShotPromptTemplate(
                examples=self.examples,
                example_prompt=self.example_prompt_template,
                prefix=prefix,
                suffix=suffix,
                input_variables=["user_question"],
                example_separator="\n\n"
            )
        
        self.chain = (
            RunnablePassthrough()
            | self.generator_prompt
            | self.llm
        )

    def generate(self, user_question: str) -> str:
        full_response = self.chain.invoke({
            "user_question": user_question
        }).content.strip()
        
        answer_match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return full_response
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate answers for Long-COVID questions')
    parser.add_argument('--question', required=True, help='Original user question')
    parser.add_argument('--context', required=True, help='Retrieved context for answering', nargs='+')
    args = parser.parse_args()
    
    full_context = "\n\n".join(args.context)
    
    generator = AnswerGeneratorRAG()
    result = generator.generate(args.question, full_context)
    print("\nGenerated Answer:")
    print(result)