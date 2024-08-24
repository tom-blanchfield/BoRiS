import streamlit as st
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to calculate alliteration score
def alliteration_score(sentence):
    words = re.findall(r'\b\w+', sentence.lower())
    if not words:
        return 0
    first_letters = [word[0] for word in words]
    letter_count = sum(first_letters.count(letter) > 1 for letter in set(first_letters))
    return letter_count / len(words)

# Simple function to tokenize text into sentences
def simple_sent_tokenize(text):
    # A basic sentence tokenizer using regular expressions
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_endings.split(text)
    return sentences

# Function to clean up sentences by removing extraneous punctuation and spaces
def clean_sentence(sentence):
    # Remove leading and trailing spaces and commas
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    sentence = re.sub(r',+', ',', sentence)  # Replace multiple commas with a single comma
    return sentence

# Function to analyze and filter sentences
def filter_sentences(sentences):
    candidate_sentences = []
    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        if sentiment_scores['compound'] > 0.5 and alliteration_score(sentence) > 0.1:
            cleaned_sentence = clean_sentence(sentence)
            candidate_sentences.append(cleaned_sentence)
    return candidate_sentences

# Streamlit App
def main():
    st.title("Dubliners Sentence Generator")

    # GitHub raw URL for the Dubliners text file
    github_raw_url = "https://raw.githubusercontent.com/yourusername/yourrepository/main/dubliners.txt"
    
    # Fetch the text file from GitHub
    response = requests.get(github_raw_url)
    if response.status_code == 200:
        text = response.text
        sentences = simple_sent_tokenize(text)
        
        st.write(f"Total Sentences: {len(sentences)}")

        # Analyze and filter sentences
        filtered_sentences = filter_sentences(sentences)

        st.write(f"Filtered Sentences Count: {len(filtered_sentences)}")
        
        if len(filtered_sentences) > 0:
            # Display a random sentence from filtered sentences
            import random
            sentence = random.choice(filtered_sentences)
            st.write("Here's a sentence with high alliteration and positive sentiment:")
            st.write(sentence)
        else:
            st.write("No sentences matched the criteria.")
    else:
        st.write("Failed to fetch the text file from GitHub.")

if __name__ == "__main__":
    main()
