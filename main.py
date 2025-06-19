from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from api.query_rewriter import QueryRewriter
from api.context_summarization import ContextSummarizer
from api.answer_generator import AnswerGenerator

import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

rewriter = QueryRewriter(model_name="gpt-4o", max_tokens=200)
context_summarizer = ContextSummarizer()
answer_generator = AnswerGenerator()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = FAISS.load_local("med_article_vdb0406", embeddings, allow_dangerous_deserialization=True)

def naive_pipeline(query: str):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    final_answer = answer_generator.generate(query, context)

    return context, final_answer

def advanced_pipeline(query: str):
    try:
        rewritten_query = rewriter.rewrite(query)
        docs = vector_store.similarity_search(rewritten_query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        summarized_context = context_summarizer.summarize_context(context)
        final_answer = answer_generator.generate(query, summarized_context)
        
        return rewritten_query, context, summarized_context, final_answer
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg

# Create RAG System as a Blocks interface
with gr.Blocks(theme=gr.themes.Soft()) as rag_tab:
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Input Query", placeholder="Enter your question...")
            btn = gr.Button("Run Pipeline", variant="primary")
        
        # with gr.Column():
            rewrite_out = gr.Textbox(label="Rewritten Query", interactive=False)
            context_out = gr.Textbox(label="Retrieved Context", lines=4, interactive=False)
            summary_out = gr.Textbox(label="Summarized Context", interactive=False)
            answer_out = gr.Textbox(label="Final Answer", interactive=False)
    
    btn.click(
        advanced_pipeline,
        inputs=inp,
        outputs=[rewrite_out, context_out, summary_out, answer_out]
    )

# Create other tabs
hello_world = gr.Interface(lambda name: "Hello " + name, "text", "text")
bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")

# Updated ChatInterface with new messages format
def chat_response(message, history):
    """Simple chat response function using new message format"""
    return f"Hello {message}"

chat = gr.ChatInterface(
    chat_response,
    examples=["What's your name?", "How does this work?"],
    title="Simple Chat"
)

# Combine into Tabbed Interface
demo = gr.TabbedInterface(
    [rag_tab, hello_world, bye_world, chat],
    ["Advanced RAG", "Hello World", "Bye World", "Chat"]
)

if __name__ == "__main__":
    demo.launch(server_port=7860)