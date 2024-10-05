import streamlit as st
from transformers import pipeline
import pandas as pd
import torch
import gc

# Task-specific functions
def perform_text_summarization():
    user_input = st.text_area("Enter the text you want to summarize:")
    if user_input:
        if st.button("Execute"):
            with st.spinner("Summarizing the text..."):
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn", trust_remote_code=True)
                summary = summarizer(user_input, max_length=100, min_length=30, do_sample=False)
                st.write(f"Summary: {summary[0]['summary_text']}")

def perform_translation():
    user_input = st.text_area("Enter the text you want to translate:")
    if user_input:
        target_language = st.selectbox("Select the target language", ["French", "Spanish", "German", "Arabic"])
        lang_codes = {"French": "fr", "Spanish": "es", "German": "de", "Arabic": "ar"}
        if st.button("Execute"):
            with st.spinner("Translating the text..."):
                translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{lang_codes[target_language]}", trust_remote_code=True)
                translation = translator(user_input)
                st.write(f"Translation: {translation[0]['translation_text']}")

def perform_text_generation():
    user_input = st.text_area("Enter the beginning of your text (the model will generate the continuation):")
    if user_input:
        if st.button("Execute"):
            with st.spinner("Generating text..."):
                text_generator = pipeline("text-generation", model="gpt2", trust_remote_code=True)
                generated_text = text_generator(user_input, max_length=100, num_return_sequences=1)
                st.write(f"Generated Text: {generated_text[0]['generated_text']}")

def perform_ner():
    user_input = st.text_area("Enter text to extract named entities:")
    if user_input:
        if st.button("Execute"):
            with st.spinner("Extracting named entities..."):
                ner_pipeline = pipeline("ner", grouped_entities=True, trust_remote_code=True)
                entities = ner_pipeline(user_input)
                st.write("Named Entities:")
                for entity in entities:
                    st.write(f"{entity['word']} ({entity['entity_group']}) - Confidence: {entity['score']:.2f}")

def perform_question_answering():
    st.write("Upload a text document (plain text only) or enter a paragraph for question answering.")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file:
        document = uploaded_file.read().decode("utf-8")
    else:
        document = st.text_area("Or, paste a paragraph:")

    if document:
        st.write("Document:")
        st.write(document)
        question = st.text_input("Enter your question:")
        if question:
            if st.button("Execute"):
                with st.spinner("Answering your question..."):
                    qa_pipeline = pipeline("question-answering", trust_remote_code=True)
                    answer = qa_pipeline(question=question, context=document)
                    st.write(f"Answer: {answer['answer']}")

# Main app logic
st.title("Text Toolkit AI")
task = st.sidebar.selectbox("Select a task", 
                            ("Text Summarization", 
                             "Translation", 
                             "Text Generation", 
                             "Named Entity Recognition", 
                             "Question Answering"))

if task == "Text Summarization":
    perform_text_summarization()
elif task == "Translation":
    perform_translation()
elif task == "Text Generation":
    perform_text_generation()
elif task == "Named Entity Recognition":
    perform_ner()
elif task == "Question Answering":
    perform_question_answering()

# Call garbage collection at the end to free up memory
gc.collect()
