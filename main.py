from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from api.query_rewriter import QueryRewriter
from api.context_summarization import ContextSummarizer
from api.answer_generator import AnswerGenerator

import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

rewriter = QueryRewriter()
context_summarizer = ContextSummarizer()
answer_generator = AnswerGenerator()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = FAISS.load_local("med_article_vdb0406", embeddings, allow_dangerous_deserialization=True)

def naive_pipeline(query: str):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([
            f"<Chunk {i+1}>\n{doc.page_content}" 
            for i, doc in enumerate(docs)
            ])

    final_answer = answer_generator.generate(query, context)

    return context, final_answer

def advanced_pipeline(query: str):
    try:
        rewritten_query = rewriter.rewrite(query)
        docs = vector_store.similarity_search(rewritten_query, k=3)
        context = "\n\n".join([
            f"<Chunk {i+1}>\n{doc.page_content}" 
            for i, doc in enumerate(docs)
            ])
        summarized_context = context_summarizer.summarize(context)
        final_answer = answer_generator.generate(query, summarized_context)
        
        return rewritten_query, context, summarized_context, final_answer
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg

inp = gr.Textbox(label="Input Query", placeholder="Enter your question...")

with gr.Blocks(theme=gr.themes.Soft()) as user_input:
    with gr.Row():
        with gr.Column():
            inp.render()
            btn = gr.Button("Run Pipeline", variant="primary")
    
    btn.click(
        inputs=inp
    )


with gr.Blocks(theme=gr.themes.Soft()) as advanced:
    with gr.Row():
        with gr.Column():
            rewrite_out = gr.Textbox(label="Rewritten Query", interactive=False)
            context_out = gr.Textbox(label="Retrieved Context", lines=4, interactive=False)
            summary_out = gr.Textbox(label="Summarized Context", interactive=False)
            answer_out = gr.Textbox(label="Final Answer", interactive=False)
    
    inp.change(
        advanced_pipeline,
        inputs=inp,
        outputs=[rewrite_out, context_out, summary_out, answer_out]
    )

with gr.Blocks(theme=gr.themes.Soft()) as naive:
    with gr.Row():
        with gr.Column():
            context_out = gr.Textbox(label="Retrieved Context", lines=4, interactive=False)
            answer_out = gr.Textbox(label="Final Answer", interactive=False)
    
    inp.change(
        lambda: gr.Textbox("Processing..."), 
    ).then(
        naive_pipeline, 
        inputs=inp, 
        outputs=[context_out, answer_out]
    )

with gr.Blocks(theme=gr.themes.Soft()) as parametric:
    with gr.Row():
        with gr.Column():
            answer_out = gr.Textbox(label="Final Answer", interactive=False)
    inp.change(
        naive_pipeline,
        inputs=inp,
        outputs=[context_out, answer_out]
    )

demo = gr.TabbedInterface(
    [user_input, advanced, naive, parametric],
    ["Ask Your Question","Advanced RAG", "Naive RAG", "Frozen LLMs"]
)

if __name__ == "__main__":
    demo.launch(server_port=7860)