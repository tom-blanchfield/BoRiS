import streamlit as st
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import random

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Simple function to tokenize text into sentences
def simple_sent_tokenize(text):
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_endings.split(text)
    return sentences

# Function to clean up sentences by removing extraneous punctuation and spaces
def clean_sentence(sentence):
    sentence = sentence.strip()
    
    # Remove multiple spaces and multiple commas
    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    sentence = re.sub(r',+', ',', sentence)  # Replace multiple commas with a single comma

    # Ensure there's a space after commas, and remove spaces before commas or periods
    sentence = re.sub(r'\s*,\s*', ', ', sentence)
    sentence = re.sub(r'\s*\.\s*', '.', sentence)

    # Remove commas before periods or other punctuation
    sentence = re.sub(r',\.', '.', sentence)
    sentence = re.sub(r',\?', '?', sentence)
    sentence = re.sub(r',!', '!', sentence)

    # Fix capitalization after periods
    sentence = re.sub(r'(?<=\.\s)([a-z])', lambda x: x.group(1).upper(), sentence)

    return sentence

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

# Function to analyze and filter sentences
def filter_sentences(sentences, alliteration_threshold, pos_threshold, neg_threshold):
    candidate_sentences = []
    failed_sentences = 0  # Track the number of failed sentences

    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        allit_score = alliteration_score(sentence)

        # Condition for selecting sentences based on alliteration, positive tone, and negative tone
        if (sentiment_scores['pos'] >= pos_threshold and 
            sentiment_scores['neg'] <= neg_threshold and 
            allit_score >= alliteration_threshold):
            
            cleaned_sentence = clean_sentence(sentence)
            candidate_sentences.append(cleaned_sentence)
        else:
            failed_sentences += 1

    # Log the number of failed sentences
    st.write(f"Number of failed sentences: {failed_sentences}")

    return candidate_sentences

# Streamlit App
def main():
    st.title("Ulysses Sentences")

    # GitHub raw URL for the Ulysses text file
    github_raw_url = "https://github.com/tom-blanchfield/BoRiS/blob/main/ulysses.txt"
    
    # Fetch the text file from GitHub
    try:
        response = requests.get(github_raw_url)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        text = response.text
        sentences = simple_sent_tokenize(text)
        
        st.write(f"Total Sentences: {len(sentences)}")

        # Add sliders for alliteration, positive tone, and negative tone thresholds
        alliteration_threshold = st.slider("Set Alliteration Threshold", 0.0, 2.0, 0.5)
        pos_threshold = st.slider("Set Minimum Positive Tone", 0.0, 1.0, 0.1)
        neg_threshold = st.slider("Set Maximum Negative Tone", 0.0, 1.0, 0.3)

        # Analyze and filter sentences based on sliders
        filtered_sentences = filter_sentences(sentences, alliteration_threshold, pos_threshold, neg_threshold)

        st.write(f"Filtered Sentences Count: {len(filtered_sentences)}")
        
        if len(filtered_sentences) > 0:
            if st.button('Generate Sentence'):
                # Display a random sentence from filtered sentences
                sentence = random.choice(filtered_sentences)
                st.write("Here's your sentence:")
                st.write(sentence)
        else:
            st.write("No sentences matched the criteria.")
    except requests.RequestException as e:
        st.write(f"Error fetching the text file: {e}")

if __name__ == "__main__":
    main()

    except requests.RequestException as e:
        st.write(f"Error fetching the text file: {e}")

if __name__ == "__main__":
    main()
