import streamlit as st
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to calculate a stricter alliteration score
def alliteration_score(sentence):
    words = re.findall(r'\b\w+', sentence.lower())
    if not words:
        return 0
    first_letters = [word[0] for word in words if word[0].isalpha()]
    alliteration_count = sum(first_letters.count(letter) for letter in set(first_letters))
    
    return alliteration_count / len(words) if len(words) > 0 else 0

# Simple function to tokenize text into sentences
def simple_sent_tokenize(text):
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_endings.split(text)
    return sentences

# Function to clean up sentences by removing extraneous punctuation and spaces
def clean_sentence(sentence):
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    sentence = re.sub(r',+', ',', sentence)  # Replace multiple commas with a single comma
    return sentence

# Function to analyze and filter sentences
def filter_sentences(sentences):
    candidate_sentences = []
    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        allit_score = alliteration_score(sentence)
        
        # Debugging output
        st.write(f"Sentence: {sentence}")
        st.write(f"Sentiment Score: {sentiment_scores['compound']}, Alliteration Score: {allit_score}")
        
        # Adjust thresholds here as needed
        if sentiment_scores['compound'] > 0.1 and allit_score > 1:
            cleaned_sentence = clean_sentence(sentence)
            candidate_sentences.append(cleaned_sentence)
    
    return candidate_sentences

# Streamlit App
def main():
    st.title("Dubliners Sentence Generator")

    # GitHub raw URL for the Dubliners text file
    github_raw_url = "https://raw.githubusercontent.com/tom-blanchfield/BoRiS/b1209408bc162dc9df3f402769bcfe63d150d5f4/dubliners.txt"
    
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
            if st.button('Generate Another Sentence'):
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
