import streamlit as st
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import random

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

# Function to tokenize text into sentences and clean up sentence fragments
def simple_sent_tokenize(text):
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_endings.split(text)
    
    # Clean up any empty or malformed sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

# Function to clean up sentences (remove extra spaces, unnecessary quotes, commas, etc.)
def clean_sentence(sentence):
    # Step 1: Strip leading and trailing spaces
    sentence = sentence.strip()
    
    # Step 2: Replace multiple spaces with a single space
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Step 3: Replace multiple commas with a single comma
    sentence = re.sub(r',+', ',', sentence)
    
    # Step 4: Remove misplaced commas (e.g., "who,was" should be "who was")
    sentence = re.sub(r'(?<=\w),(?=\w)', ', ', sentence)
    
    # Step 5: Remove leading or trailing commas
    sentence = re.sub(r'^,|,$', '', sentence)
    
    # Step 6: Remove stray quotes
    sentence = re.sub(r'^"|"$', '', sentence)  # Remove quotes at the start/end
    sentence = re.sub(r'"\s*$', '', sentence)  # Remove trailing quotes after periods
    sentence = re.sub(r'\s*"\s*', '', sentence)  # Remove any remaining stray quotes
    
    return sentence

# Function to analyze and filter sentences
def filter_sentences(sentences, sentiment_threshold, alliteration_threshold, pos_threshold, neg_threshold, neu_threshold):
    candidate_sentences = []
    failed_sentences = []

    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        allit_score = alliteration_score(sentence)

        # Stricter condition: Require both a minimum sentiment threshold and an alliteration threshold
        if (sentiment_scores['compound'] >= sentiment_threshold and  # Compound sentiment must be high
            sentiment_scores['pos'] >= pos_threshold and  # Positive tone must be above threshold
            allit_score >= alliteration_threshold and  # Alliteration must be above threshold
            sentiment_scores['neg'] <= neg_threshold):  # Negative sentiment should be below threshold
            
            cleaned_sentence = clean_sentence(sentence)
            candidate_sentences.append(cleaned_sentence)
        else:
            failed_sentences.append({
                'sentence': sentence,
                'compound': sentiment_scores['compound'],
                'pos': sentiment_scores['pos'],
                'neg': sentiment_scores['neg'],
                'neu': sentiment_scores['neu'],
                'alliteration': allit_score
            })

    # Debugging output: Check why sentences were rejected
    st.write("Debug Info: Failed Sentences")
    for fs in failed_sentences:
        st.write(f"Sentence: {fs['sentence']}, Compound: {fs['compound']}, Pos: {fs['pos']}, Neg: {fs['neg']}, Neu: {fs['neu']}, Alliteration: {fs['alliteration']}")

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

        # Add sliders for sentiment and alliteration thresholds
        sentiment_threshold = st.slider("Set Sentiment Threshold (Compound)", -1.0, 1.0, 0.5)  # Start at 0.5 for positivity
        alliteration_threshold = st.slider("Set Alliteration Threshold", 0.0, 2.0, 0.8)  # Require high alliteration

        # Additional sliders for tone adjustment
        pos_threshold = st.slider("Set Minimum Positive Tone", 0.0, 1.0, 0.3)  # Require some positivity
        neg_threshold = st.slider("Set Maximum Negative Tone", 0.0, 1.0, 0.2)  # Keep negativity low
        neu_threshold = st.slider("Set Minimum Neutral Tone", 0.0, 1.0, 0.0)  # Keep neutral tone flexible

        # Analyze and filter sentences based on sliders
        filtered_sentences = filter_sentences(sentences, sentiment_threshold, alliteration_threshold, pos_threshold, neg_threshold, neu_threshold)

        st.write(f"Filtered Sentences Count: {len(filtered_sentences)}")
        
        if len(filtered_sentences) > 0:
            if st.button('Generate Sentence'):
                # Display a random sentence from filtered sentences
                sentence = random.choice(filtered_sentences)
                st.write("Here's a sentence with high alliteration and positive sentiment:")
                st.write(sentence)
        else:
            st.write("No sentences matched the criteria.")
    except requests.RequestException as e:
        st.write(f"Error fetching the text file: {e}")

if __name__ == "__main__":
    main()

