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
    first_letters = [word[0] for word in words if word[0].isalpha()]
    alliteration_count = sum(first_letters.count(letter) for letter in set(first_letters))
    if len(words) > 2:
        return alliteration_count / len(words)
    return 0

# Function to tokenize text into sentences (basic)
def simple_sent_tokenize(text):
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_endings.split(text)
    return sentences

# Function to clean up sentences
def clean_sentence(sentence):
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    sentence = re.sub(r',+', ',', sentence)  # Replace multiple commas with a single comma
    return sentence

# Function to analyze and filter sentences
def filter_sentences(sentences, sentiment_threshold, alliteration_threshold, pos_threshold, neg_threshold, neu_threshold):
    candidate_sentences = []
    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        
        # Apply multiple tone filters based on slider values
        if (sentiment_scores['compound'] > sentiment_threshold and 
            alliteration_score(sentence) > alliteration_threshold and
            sentiment_scores['pos'] >= pos_threshold and
            sentiment_scores['neg'] <= neg_threshold and
            sentiment_scores['neu'] >= neu_threshold):
            
            cleaned_sentence = clean_sentence(sentence)
            candidate_sentences.append(cleaned_sentence)
            
    return candidate_sentences

# Streamlit App
def main():
    st.title("Ulysses Alliterative Question Generator")

    # GitHub raw URL for the Ulysses text file
    github_raw_url = "https://raw.githubusercontent.com/your-username/your-repo/main/ulysses.txt"
    
    # Fetch the text file from GitHub
    response = requests.get(github_raw_url)
    if response.status_code == 200:
        text = response.text
        sentences = simple_sent_tokenize(text)
        
        st.write(f"Total Sentences: {len(sentences)}")

        # Add sliders for sentiment and alliteration thresholds
        sentiment_threshold = st.slider("Set Sentiment Threshold (Compound)", 0.0, 1.0, 0.5)
        alliteration_threshold = st.slider("Set Alliteration Threshold", 0.0, 2.0, 1.0)

        # Additional sliders for tone adjustment
        pos_threshold = st.slider("Set Minimum Positive Tone", 0.0, 1.0, 0.1)
        neg_threshold = st.slider("Set Maximum Negative Tone", 0.0, 1.0, 0.2)
        neu_threshold = st.slider("Set Minimum Neutral Tone", 0.0, 1.0, 0.5)

        # Analyze and filter sentences based on sliders
        filtered_sentences = filter_sentences(sentences, sentiment_threshold, alliteration_threshold, pos_threshold, neg_threshold, neu_threshold)

        st.write(f"Filtered Sentences Count: {len(filtered_sentences)}")
        
        if len(filtered_sentences) > 0:
            if st.button('Generate Alliterative Question'):
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
