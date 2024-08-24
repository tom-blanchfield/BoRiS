import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Download necessary NLTK resources
nltk.download('punkt')

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

# Function to analyze and filter sentences
def filter_sentences(sentences):
    candidate_sentences = []
    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        if sentiment_scores['compound'] > 0.5 and alliteration_score(sentence) > 0.1:
            candidate_sentences.append(sentence)
    return candidate_sentences

# Streamlit App
def main():
    st.title("Dubliners Sentence Generator")

    # Upload the text file
    uploaded_file = st.file_uploader("Upload the Dubliners text file", type=["txt"])
    
    if uploaded_file is not None:
        text = uploaded_file.read().decode('utf-8')
        sentences = sent_tokenize(text)
        
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

if __name__ == "__main__":
    main()
