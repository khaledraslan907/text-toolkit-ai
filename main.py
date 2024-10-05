import streamlit as st
from transformers import pipeline
from textblob import TextBlob
from summarizer import Summarizer

# Language codes for translation
lang_codes = {
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl"
}

# Function for sentiment analysis
def perform_sentiment_analysis(input_text):
    blob = TextBlob(input_text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive", sentiment_score
    elif sentiment_score == 0:
        return "Neutral", sentiment_score
    else:
        return "Negative", sentiment_score

# Function for text summarization
def perform_summarization(input_text):
    model = Summarizer()
    summary = model(input_text, num_sentences=3)
    return summary

# Function for translation with error handling
def perform_translation(input_text, target_language):
    try:
        model_name = f"Helsinki-NLP/opus-mt-en-{lang_codes[target_language]}"
        translator = pipeline("translation", model=model_name, trust_remote_code=True)
        translated_text = translator(input_text)[0]['translation_text']
        return translated_text
    except ValueError as ve:
        st.error(f"Error: {ve}")
    except KeyError:
        st.error("The selected target language is not supported.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Streamlit UI
st.title("Multitasking AI Application")

# Sidebar for navigation
st.sidebar.title("Tasks")
task = st.sidebar.radio("Choose a task:", ("Sentiment Analysis", "Text Summarization", "Translation"))

# Input text area
input_text = st.text_area("Enter your text here:")

# Execution button
execute_button = st.button("Execute Task")

# Sentiment Analysis
if task == "Sentiment Analysis":
    if execute_button:
        if input_text:
            sentiment, score = perform_sentiment_analysis(input_text)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Sentiment Score: {score}")
        else:
            st.error("Please enter some text for sentiment analysis.")

# Text Summarization
if task == "Text Summarization":
    if execute_button:
        if input_text:
            summary = perform_summarization(input_text)
            st.write("Summary:")
            st.write(summary)
        else:
            st.error("Please enter some text for summarization.")

# Translation
if task == "Translation":
    target_language = st.selectbox("Choose target language:", list(lang_codes.keys()))
    
    if execute_button:
        if input_text:
            translated_text = perform_translation(input_text, target_language)
            if translated_text:
                st.write("Translated Text:")
                st.write(translated_text)
        else:
            st.error("Please enter text to translate.")
