import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import NMF
from surprise import SVD, KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
import streamlit as st
import altair as alt

# Load the data
ratings = pd.read_csv('ratings.csv')

# Ensure the ratings DataFrame has the required columns
required_columns = ['user_id', 'book_id', 'rating']
missing_columns = [col for col in required_columns if col not in ratings.columns]

if missing_columns:
    st.error(f"The ratings DataFrame is missing the following required columns: {', '.join(missing_columns)}")
    st.stop()

# Calculate the number of ratings per user
user_ratings_count = ratings['user_id'].value_counts()

# Get the top 1,000 users by number of ratings
top_users = user_ratings_count.nlargest(5000).index

# Filter ratings for the top users
ratings = ratings[ratings['user_id'].isin(top_users)]

# Filter relevant columns
ratings = ratings[['user_id', 'book_id', 'rating']]

# Create a pivot table of user ratings
user_ratings_pivot = ratings.pivot_table(index='user_id', columns='book_id', values='rating', fill_value=0)

# List of algorithms to test
algorithms = [
    'Cosine Similarity',
    'Adjusted Cosine Similarity',
    'Jaccard',
    'Euclidean Distance',
    'Non-Zero Matrix Factorisation'
]

# Function to calculate algorithm similarity
def calculate_similarity(algorithm):
    similarity_score = None  # Initialise the similarity_score variable

    if algorithm == 'Cosine Similarity':
        st.text("Calculating Cosine Similarities...")
        # Get the unique book IDs from train_set
        train_set_books = user_ratings_pivot.columns

        # Calculate similarity using the user_ratings_pivot pivot table
        similarities = cosine_similarity(user_ratings_pivot.T)
        similarity_score = np.mean(similarities)
        st.text(f"Cosine Similarities calculation complete.\nSimilarity Score: {similarity_score:.4f}")
        st.text("Cosine Similarity measures the similarity between users based on their \nrating patterns. Higher scores indicate more similar users.")

    elif algorithm == 'Adjusted Cosine Similarity':
        st.text("Calculating Adjusted Cosine Similarities...")
        # Calculate mean rating for each book
        book_means = user_ratings_pivot.mean(axis=0)

        # Subtract mean rating from each user's rating
        ratings_centered = user_ratings_pivot.sub(book_means, axis=1)

        # Calculate similarity using the centered ratings
        similarities = cosine_similarity(ratings_centered.T)
        similarity_score = np.mean(similarities)
        st.text(f"Adjusted Cosine Similarities calculation complete.\nSimilarity Score: {similarity_score:.4f}")
        st.text("Adjusted Cosine Similarity measures the similarity between users based \non their rating patterns, while considering the average rating of each book.\nHigher scores indicate more similar users.")

    elif algorithm == 'Jaccard':
        st.text("Calculating Jaccard Similarities...")
        # Convert DataFrame to NumPy array
        user_ratings_array = user_ratings_pivot.values

        # Calculate Jaccard similarity
        similarities = pairwise_distances(user_ratings_array, metric='jaccard')
        similarity_score = np.mean(1 - similarities)
        st.text(f"Jaccard Similarities calculation complete.\nSimilarity Score: {similarity_score:.4f}")
        st.text("Jaccard Similarity measures the similarity between users based on the \npresence or absence of ratings.\nHigher scores indicate more similar users.")

    elif algorithm == 'Euclidean Distance':
        st.text("Calculating Euclidean Distances...")
        # Calculate Euclidean distance
        similarities = pairwise_distances(user_ratings_pivot, metric='euclidean')
        similarity_score = np.mean(1 / (1 + similarities))
        st.text(f"Euclidean Distances calculation complete.\nSimilarity Score: {similarity_score:.4f}")
        st.text("Euclidean Distance measures the dissimilarity between users based on their \nrating patterns. Lower scores indicate more similar users.")

    elif algorithm == 'Non-Zero Matrix Factorisation':
        st.text("Calculating Non-Zero Matrix Factorsation Similarities...")
        # Implement the NMF algorithm
        nmf = NMF(n_components=10)
        nmf.fit(user_ratings_pivot)
        user_features = nmf.transform(user_ratings_pivot)
        item_features = nmf.components_

        # Transpose either user_features or item_features
        # to ensure compatibility for cosine similarity calculation
        similarities = cosine_similarity(user_features, item_features.T)
        similarity_score = np.mean(similarities)
        st.text(f"Non-Zero Matrix Factorisation Similarities calculation complete.\nSimilarity Score: {similarity_score:.4f}")
        st.text("Non-Zero Matrix Factorisation calculates the similarity between users based on a \nmatrix factorization approach.\nHigher scores indicate more similar users.")

# Streamlit app
st.title("Recommender Algorithm Evaluator")
st.sidebar.title("Options")

if st.sidebar.button("Test Algorithms"):
    results = {}  # Dictionary to store the results of each algorithm
    for algorithm in algorithms:
        similarity_score = calculate_similarity(algorithm)  # Assign the similarity score to a variable
        results[algorithm] = similarity_score  # Store the similarity score



 
