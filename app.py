import torch
import wikipedia
import transformers
import streamlit as st
from tokenizers import Tokenizer
from transformers import pipeline, Pipeline 
import pywikibot


pywikibot.config.socket_timeout = 60  # Set timeout to 60 seconds

@st.cache(allow_output_mutation=True)
def load_qa_pipeline() -> pipeline:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline

@st.cache_data
def load_wiki_summary(topic: str) -> str:
    try:
        site = pywikibot.Site("en", "wikipedia")
        page = pywikibot.Page(site, topic)
        if page.isRedirectPage():
            return f"Redirected page: {page.title()}"
        if not page.exists():
            return "Page not found."
        return page.text[:3000]
    except Exception as e:
        return f"Error: {e}"

def answer_question(pipeline: pipeline, question: str, paragraph: str) -> dict:
    input_data = {
        "question": question,
        "context": paragraph
    }
    output = pipeline(input_data, truncation=True, padding=True)
    return output


if __name__ == "__main__":
    st.title("Wikipedia Question Answering")
    st.write("Search topic, Ask questions, Get Answers")

    topic = st.text_input("SEARCH TOPIC", "")
    article_paragraph = st.empty()
    question = st.text_input("QUESTION", "")

    if topic:
        summary = load_wiki_summary(topic)
        article_paragraph.markdown(summary)

        if question.strip():
            qa_pipeline = load_qa_pipeline()
            result = answer_question(qa_pipeline, question, summary)
            answer = result["answer"]
            st.write(answer)