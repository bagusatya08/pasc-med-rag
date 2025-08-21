import pandas as pd
import os
import time
import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.query_rewriter import QueryRewriter
from src.context_summarization import ContextSummarizer
from src.answer_generator import AnswerGeneratorRAG, AnswerGeneratorFrozen
import gradio as gr

def initialize_dataframe():
    """
    Initializes a pandas DataFrame to store results.
    If a CSV file with previous results exists, it loads it.
    Otherwise, it creates an empty DataFrame with the specified columns.
    """
    columns = [
        "timestamp", "pipeline_type", "query", "rewritten_query",
        "retrieved_context", "summarized_context", "thinking_process", "final_answer",
        "source_1", "score_1", "source_2", "score_2", "source_3", "score_3",
        "rewriting_time_s", "retrieval_time_s", "summarization_time_s", "generation_time_s"
    ]

def log_results(data_dict):
    """
    Appends a new row of results to the global DataFrame and saves it to a CSV file.
    
    Args:
        data_dict (dict): A dictionary containing the data for the new row.
    """
    global results_df

    new_row = pd.DataFrame([data_dict])
    results_df = pd.concat([results_df, new_row], ignore_index=True)


results_df = initialize_dataframe()

rewriter = QueryRewriter()
context_summarizer = ContextSummarizer()
answer_generator_RAG = AnswerGeneratorRAG()
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

vector_store = FAISS.load_local("med_article_vdb2706", embeddings, allow_dangerous_deserialization=True)
answer_generator_frozen = AnswerGeneratorFrozen()


def naive_pipeline(query: str):
    timings = {}
    start_time = time.perf_counter()
    docs = vector_store.similarity_search_with_score(query, k=3)
    end_time = time.perf_counter()
    timings['retrieval'] = end_time - start_time

    source_links = []
    scores = []
    formatted_chunks = []
    for i, (doc, score) in enumerate(docs):
        chunk_text = f"<Chunk {i+1}>\n{doc.page_content}"
        formatted_chunks.append(chunk_text)
        source = doc.metadata["source"].replace("gs:/", "https://storage.googleapis.com")
        source_links.append(source)
        scores.append(score)

    context = "\n\n".join(formatted_chunks)
    start_time = time.perf_counter()
    final_answer = answer_generator_RAG.generate(query, context)
    end_time = time.perf_counter()
    timings['answer_generation'] = end_time - start_time
    
    return context, final_answer, source_links, scores, timings

def advanced_pipeline(query: str):
    timings = {}
    start_time = time.perf_counter()
    rewritten_query = rewriter.rewrite(query)
    end_time = time.perf_counter()
    timings['query_rewriting'] = end_time - start_time

    start_time = time.perf_counter()
    docs = vector_store.similarity_search_with_score(rewritten_query, k=3)
    end_time = time.perf_counter()
    timings['retrieval'] = end_time - start_time

    source_links, scores, formatted_chunks = [], [], []
    for i, (doc, score) in enumerate(docs):
        chunk_text = f"<Chunk {i+1}>\n{doc.page_content}"
        formatted_chunks.append(chunk_text)
        source = doc.metadata["source"].replace("gs:/", "https://storage.googleapis.com")
        source_links.append(source)
        scores.append(score)

    context = "\n\n".join(formatted_chunks)
    start_time = time.perf_counter()
    summarized_context = context_summarizer.summarize(context)
    end_time = time.perf_counter()
    timings['summarization'] = end_time - start_time

    start_time = time.perf_counter()
    final_answer = answer_generator_RAG.generate(rewritten_query, summarized_context)
    end_time = time.perf_counter()
    timings['generation'] = end_time - start_time
    
    return rewritten_query, context, summarized_context, final_answer, source_links, scores, timings

def frozen_pipeline(query: str):
    timings = {}
    start_time = time.perf_counter()
    final_answer = answer_generator_frozen.generate(query)
    end_time = time.perf_counter()
    timings['generation'] = end_time - start_time
    return final_answer, timings


def ui_naive(query):
    context, answer, source_links, scores, timings = naive_pipeline(query)

    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_type": "Naive RAG",
        "query": query,
        "retrieved_context": context,
        "retrieval_time_s": timings.get("retrieval"),
        "generation_time_s": timings.get("answer_generation"),
    }
    
    parts = answer.split("</think>")
    if len(parts) == 2:
        log_data["thinking_process"] = parts[0].replace("<think>", "").strip()
        log_data["final_answer"] = parts[1].strip()
    else:
        log_data["thinking_process"] = ""
        log_data["final_answer"] = answer

    for i in range(3):
        log_data[f"source_{i+1}"] = source_links[i] if i < len(source_links) else None
        log_data[f"score_{i+1}"] = scores[i] if i < len(scores) else None

    log_results(log_data)

    source_updates, score_updates = [], []
    for i in range(3):
        if i < len(source_links):
            source_updates.append(gr.Button(value=f"Source {i+1}", link=source_links[i], visible=True))
            score_updates.append(gr.Textbox(value=f"{scores[i]:.4f}", visible=True))
        else:
            source_updates.append(gr.Button(visible=False))
            score_updates.append(gr.Textbox(visible=False))
            
    return (
        context, log_data["thinking_process"], log_data["final_answer"],
        *source_updates, *score_updates,
        f"{timings['retrieval']:.2f} seconds",
        f"{timings['answer_generation']:.2f} seconds"
    )

def ui_advanced(query):
    rewritten_query, context, summarized_context, answer, source_links, scores, timings = advanced_pipeline(query)

    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_type": "Advanced RAG",
        "query": query,
        "rewritten_query": rewritten_query,
        "retrieved_context": context,
        "summarized_context": summarized_context,
        "rewriting_time_s": timings.get("query_rewriting"),
        "retrieval_time_s": timings.get("retrieval"),
        "summarization_time_s": timings.get("summarization"),
        "generation_time_s": timings.get("generation"),
    }

    parts = answer.split("</think>")
    if len(parts) == 2:
        log_data["thinking_process"] = parts[0].replace("<think>", "").strip()
        log_data["final_answer"] = parts[1].strip()
    else:
        log_data["thinking_process"] = ""
        log_data["final_answer"] = answer
        
    for i in range(3):
        log_data[f"source_{i+1}"] = source_links[i] if i < len(source_links) else None
        log_data[f"score_{i+1}"] = scores[i] if i < len(scores) else None

    log_results(log_data)
    
    source_updates, score_updates = [], []
    for i in range(3):
        if i < len(source_links):
            source_updates.append(gr.Button(value=f"Source {i+1}", link=source_links[i], visible=True))
            score_updates.append(gr.Textbox(value=f"{scores[i]:.4f}", visible=True))
        else:
            source_updates.append(gr.Button(visible=False))
            score_updates.append(gr.Textbox(visible=False))

    return (
        rewritten_query, context, summarized_context, log_data["thinking_process"], log_data["final_answer"],
        *source_updates, *score_updates,
        f"{timings['query_rewriting']:.2f} seconds",
        f"{timings['retrieval']:.2f} seconds",
        f"{timings['summarization']:.2f} seconds",
        f"{timings['generation']:.2f} seconds"
    )

def ui_frozen(query):
    answer, timings = frozen_pipeline(query)

    log_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_type": "Frozen LLM",
        "query": query,
        "generation_time_s": timings.get("generation"),
    }
    
    parts = answer.split("</think>")
    if len(parts) == 2:
        log_data["thinking_process"] = parts[0].replace("<think>", "").strip()
        log_data["final_answer"] = parts[1].strip()
    else:
        log_data["thinking_process"] = ""
        log_data["final_answer"] = answer
        
    log_results(log_data)

    return log_data["thinking_process"], log_data["final_answer"], f"{timings['generation']:.2f} seconds"

with gr.Blocks(theme=gr.themes.Origin()) as demo:
    with gr.Tab("User Input"):
        inp = gr.Textbox(label="Input Query", placeholder="Enter your medical question here...", scale=4)

    with gr.Tab("Advanced RAG"):
        with gr.Row():
            with gr.Column(scale=1):
                adv_source1 = gr.Button(value="Source 1")
                adv_score1 = gr.Textbox(label="Score 1", interactive=False)
                adv_source2 = gr.Button(value="Source 2")
                adv_score2 = gr.Textbox(label="Score 2", interactive=False)
                adv_source3 = gr.Button(value="Source 3")
                adv_score3 = gr.Textbox(label="Score 3", interactive=False)
            with gr.Column(scale=3):
                adv_rewrite_out = gr.Textbox(label="Rewritten Query", interactive=False, lines=2)
                adv_context_out = gr.Textbox(label="Retrieved Context", lines=8, interactive=False)
                adv_summary_out = gr.Textbox(label="Summarized Context", interactive=False, lines=4)
        with gr.Accordion("Thinking Process & Timings", open=False):
            adv_think_out = gr.Textbox(label="Thinking Process", lines=8, interactive=False)
            with gr.Row():
                adv_rewriting_time = gr.Textbox(label="Rewriting Time", interactive=False)
                adv_retrieval_time = gr.Textbox(label="Retrieval Time", interactive=False)
                adv_summarization_time = gr.Textbox(label="Summarization Time", interactive=False)
                adv_generation_time = gr.Textbox(label="Generation Time", interactive=False)
        adv_answer_out = gr.Textbox(label="Final Answer", lines=5, interactive=False)
        
    with gr.Tab("Naive RAG"):
        with gr.Row():
            with gr.Column(scale=1):
                naive_source1 = gr.Button(value="Source 1")
                naive_score1 = gr.Textbox(label="Score 1", interactive=False)
                naive_source2 = gr.Button(value="Source 2")
                naive_score2 = gr.Textbox(label="Score 2", interactive=False)
                naive_source3 = gr.Button(value="Source 3")
                naive_score3 = gr.Textbox(label="Score 3", interactive=False)
            with gr.Column(scale=3):
                naive_context_out = gr.Textbox(label="Retrieved Context", lines=15, interactive=False)
        with gr.Accordion("Thinking Process & Timings", open=False):
            naive_think_out = gr.Textbox(label="Thinking Process", lines=8, interactive=False)
            with gr.Row():
                naive_retrieval_time = gr.Textbox(label="Retrieval Time", interactive=False)
                naive_generation_time = gr.Textbox(label="Generation Time", interactive=False)
        naive_answer_out = gr.Textbox(label="Final Answer", lines=5, interactive=False)

    with gr.Tab("Frozen LLM"):
        with gr.Accordion("Thinking Process & Timings", open=False):
            frozen_think_out = gr.Textbox(label="Thinking Process", lines=8, interactive=False)
            frozen_generation_time = gr.Textbox(label="Generation Time", interactive=False)
        frozen_answer_out = gr.Textbox(label="Final Answer", lines=5, interactive=False)

    inp.submit(
        ui_advanced,
        inputs=inp,
        outputs=[
            adv_rewrite_out, adv_context_out, adv_summary_out, adv_think_out, adv_answer_out,
            adv_source1, adv_source2, adv_source3,
            adv_score1, adv_score2, adv_score3,
            adv_rewriting_time, adv_retrieval_time, adv_summarization_time, adv_generation_time
        ]
    )
    inp.submit(
        ui_naive,
        inputs=inp,
        outputs=[
            naive_context_out, naive_think_out, naive_answer_out, 
            naive_source1, naive_source2, naive_source3,
            naive_score1, naive_score2, naive_score3,
            naive_retrieval_time, naive_generation_time
        ]
    )
    inp.submit(
        ui_frozen,
        inputs=inp,
        outputs=[
            frozen_think_out, frozen_answer_out, frozen_generation_time
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True)