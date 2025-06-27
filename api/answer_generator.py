from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import  ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv
import regex as re

import argparse
import sys

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

class AnswerGeneratorRAG:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_base="https://router.huggingface.co/novita/v3/openai",
            openai_api_key=HF_TOKEN,
            temperature=0.4,
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
        You are a medical expert specialized in Long-COVID (PASC). Provide accurate, evidence-based answers using ONLY the provided context.
        You need to act as a Doctor/Clinician communicating with a patient.
        </role>

        <reasoning steps>
        Before generating your answer, follow this reasoning process:
        1. **Understand the Question**: Identify key medical entities and patient context
        2. **Context Analysis**: 
            - What context chunks are relevant to the question?
            - What specific evidence supports the answer?
            - Are there conflicting information between chunks?
        3. **Knowledge Gap Check**:
            - Is sufficient information available in the context?
            - What critical information is missing?
        4. **Safety Consideration**:
            - Are there any risk factors or red flags mentioned?
            - Does the answer require professional consultation?
        5. **Response Structuring**:
            - How to present complex medical information simply?
            - What analogies or comparisons would help understanding?
        </reasoning steps>
        
        <tasks>
        1. **Generate the answer** based on these guidelines:  
        • Use ONLY information from the provided context
        • If answer isn't in context, say "I don't have enough information"
        • Use plain language suitable for patients
        • Be empathetic and supportive
        • Structure response clearly with bullet points where helpful
        • Include important medical details but explain them simply

        2. **Safety and Ethics**:
        • Never provide medical advice beyond general information
        • Always recommend consulting a healthcare provider
        • Acknowledge limitations of the information
        </tasks>
        
        Here are some examples:
        """
        
        suffix = """
        Now generate an answer for this question using the provided context:
        <input question>
        {user_question}
        </input question>
        
        <context>
        {context}
        </context>

        <output format>
        Follow this exact output format:
        
        <reasoning>
        [Your step-by-step reasoning here]
        </reasoning>
        
        <answer>
        [Your final answer here]
        </answer>
        </output format>
        """
        
        self.examples = [
            {
                "user_question": "Why am I so tired months after COVID?",
                "context": """
                <Chunk 1>
                Post-COVID fatigue syndrome (PCFS) affects 58-80% of Long-COVID patients. It's characterized by persistent exhaustion lasting >6 months after acute infection. Key features include post-exertional malaise (PEM) and non-restorative sleep.
                
                <Chunk 2>
                Underlying mechanisms may include mitochondrial dysfunction, persistent inflammation, and autonomic nervous system dysregulation. A 2023 NIH study found abnormal energy metabolism in 72% of Long-COVID fatigue patients.
                
                <Chunk 3>
                Management strategies: 1) Activity pacing 2) Sleep hygiene 3) Graded exercise therapy (GET) under supervision 4) Cognitive behavioral therapy for fatigue (CBT-F). Most patients show improvement within 6-12 months with proper management.
                """,
                "reasoning": """
                1. Key question elements: persistent fatigue (>months), post-COVID context
                2. Context relevance:
                    - Chunk 1: Confirms PCFS prevalence and duration (>6 months)
                    - Chunk 2: Explains biological mechanisms (mitochondrial, inflammation, autonomic)
                    - Chunk 3: Provides management strategies and prognosis
                3. No conflicts between chunks - all complementary
                4. Missing: Patient's specific duration, severity, comorbidities
                5. Safety: No red flags but recommend professional consultation
                6. Structure: Explain causes simply, list management options, provide hope
                """,
                "answer": """
                I understand you're experiencing persistent fatigue after COVID, which is very common. Based on current medical understanding:

                • This is known as Post-COVID Fatigue Syndrome (PCFS), affecting 58-80% of Long-COVID patients
                • The fatigue typically lasts more than 6 months after initial infection
                • Possible causes include:
                    - Energy production issues in cells
                    - Ongoing inflammation
                    - Nervous system regulation problems
                
                Recommended management approaches:
                ✓ Balance activity and rest (pacing)
                ✓ Improve sleep habits
                ✓ Gradual exercise program (with professional guidance)
                ✓ Cognitive therapy for fatigue management
                
                Most patients see improvement within 6-12 months with proper management. I recommend discussing these options with your healthcare provider to develop a personalized plan.
                """
            },
            {
                "user_question": "What can I do about brain fog after COVID?",
                "context": """
                <Chunk 1>
                Cognitive impairment ("brain fog") affects 32-56% of Long-COVID patients. Symptoms include memory problems, difficulty concentrating, and slowed processing speed. These often persist for 6+ months.
                
                <Chunk 2>
                2024 clinical trial (n=214) showed computerized cognitive training improved processing speed by 32% and working memory by 28% after 12 weeks. Mindfulness-based interventions also showed benefit.
                
                <Chunk 3>
                Pharmacological options are limited. Some evidence for off-label use of stimulants (methylphenidate) in severe cases, but more research needed. Always consult neurologist before medication.
                """,
                "reasoning": """
                1. Key question: Brain fog management strategies
                2. Context analysis:
                   - Chunk 1: Establishes prevalence and symptom profile
                   - Chunk 2: Shows efficacy of cognitive training (32% improvement) and mindfulness
                   - Chunk 3: Notes medication limitations and need for specialist consultation
                3. Missing: Patient's specific symptoms duration and severity
                4. Safety: Medications require specialist supervision
                5. Structure: Present non-drug options first, mention medications as last resort
                """,
                "answer": """
                Brain fog is a common and challenging aspect of Long-COVID. Based on current evidence:

                • It affects 32-56% of patients and often lasts 6+ months
                • Effective non-medical approaches include:
                  - Computerized brain training (shown to improve thinking speed by 32%)
                  - Mindfulness and meditation techniques
                  - Cognitive rehabilitation therapy
                
                Medication options are currently limited and should only be considered:
                - For severe cases
                - Under neurological supervision
                - When non-drug approaches haven't helped

                I recommend starting with brain training exercises and mindfulness techniques. If symptoms significantly impact your daily life, consider asking your doctor for a referral to a neurologist or cognitive specialist.
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
        You are a medical expert specialized in Long-COVID (PASC).
        You need to act as a Doctor/Clinician communicating with a patient.
        </role>

        <reasoning steps>
        Before generating your answer, follow this reasoning process:
        1. **Understand the Question**: Identify key medical entities
        2. **Safety Consideration**:
            - Are there any risk factors or red flags mentioned?
            - Does the answer require professional consultation?
        3. **Response Structuring**:
            - How to present complex medical information simply?
            - What analogies or comparisons would help understanding?
        </reasoning steps>
        
        <tasks>
        1. **Safety and Ethics**:
        • Never provide medical advice beyond general information
        • Always recommend consulting a healthcare provider
        • Acknowledge limitations of the information
        </tasks>
        
        Here are some examples:
        """
        
        suffix = """
        Now generate an answer for this question:
        <input question>
        {user_question}
        </input question>

        <output format>
        Follow this exact output format:
        
        <reasoning>
        [Your step-by-step reasoning here]
        </reasoning>
        
        <answer>
        [Your final answer here]
        </answer>
        </output format>
        """
        
        self.examples = [
            {
                "user_question": "Why am I so tired months after COVID?",
                "reasoning": """
                1. Key question elements: persistent fatigue (>months), post-COVID context
                2. Safety: No red flags but recommend professional consultation
                3. Structure: Explain causes simply, list management options, provide hope
                """,
                "answer": """
                I understand you're experiencing persistent fatigue after COVID, which is very common. Based on current medical understanding:

                • This is known as Post-COVID Fatigue Syndrome (PCFS), affecting 58-80% of Long-COVID patients
                • The fatigue typically lasts more than 6 months after initial infection
                • Possible causes include:
                    - Energy production issues in cells
                    - Ongoing inflammation
                    - Nervous system regulation problems
                
                Recommended management approaches:
                ✓ Balance activity and rest (pacing)
                ✓ Improve sleep habits
                ✓ Gradual exercise program (with professional guidance)
                ✓ Cognitive therapy for fatigue management
                
                Most patients see improvement within 6-12 months with proper management. I recommend discussing these options with your healthcare provider to develop a personalized plan.
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