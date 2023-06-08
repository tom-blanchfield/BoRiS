import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
import uuid
from PIL import Image

# Load the data
books = pd.read_csv('books.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')

# Merge tags data
book_data = pd.merge(books, book_tags, on='goodreads_book_id')
book_data = pd.merge(book_data, tags, on='tag_id')

# Get top 1000 tags
tags_df = book_data.groupby('tag_name').size().reset_index(name='counts').sort_values('counts', ascending=False).head(1000)

# Tags to include in the "genres" multi-select dropdown
tags_to_include = ['young-adult', 'literature', 'romance', 'mystery', 'science-fiction', 'fantasy', 'horror', 'thriller', 'western', 'dystopian', 'memoir', 'biography', 'autobiography', 'history', 'travel', 'cookbook', 'self-help', 'business', 'finance', 'psychology', 'philosophy', 'religion', 'art', 'music', 'comics', 'graphic novels', 'poetry', 'sport', 'humorous', 'war', 'funny']

# Title
st.sidebar.title("Please choose your favourite authors, and or genres")

# Allow the user to select multiple authors
selected_authors = st.sidebar.multiselect("Select authors", books['authors'])

#Allow the user to select multiple genres
selected_tags = st.sidebar.multiselect("Select genres", tags_to_include)

# Modify the filtered data based on the selected authors
filtered_data = book_data[book_data['authors'].isin(selected_authors) | book_data['tag_name'].isin(selected_tags)]

# Group by book and sort by count
grouped_data = filtered_data.groupby('title')['count'].sum().sort_values(ascending=False)

# Get top 10,000 raters
ratings_count = ratings.groupby('user_id').size().reset_index(name='count').sort_values('count', ascending=False)
top_raters = ratings_count[:1000]['user_id'].tolist()

# Create a DataFrame to store user ratings
user_ratings = pd.DataFrame(columns=['book_id', 'user_id', 'rating'])

# Display books to rate
st.title("Please rate these books:")
if len(grouped_data) == 0:
    st.write("No books found with selected authors or genres")
else:
    # Create three columns for book ratings
    columns = st.columns(3)
    count = 0

    for title, count in grouped_data[:9].items():
        # Get the book ID and image URL
        book_id = books.loc[books['title'] == title, 'book_id'].values[0]
        image_url = books.loc[books['title'] == title, 'image_url'].values[0]
        # Download the image from the URL
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)

            # Adjust the image size
            image = image.resize((200, 300))

            # Display the image with the title
            columns[count % 3].image(image, use_column_width=False, width=200)
        except (requests.HTTPError, OSError) as e:
            st.write(f"Error downloading image for book: {title}")

    # Get user rating for the book
    rating = st.selectbox(f"Rate the book '{title}'", options=[0, 1, 2, 3, 4, 5])

    # Add the rating to the user_ratings DataFrame
    user_ratings = user_ratings.append(pd.Series({'book_id': book_id, 'user_id': 'user1', 'rating': rating}), ignore_index=True)


# Generate recommendations based on user ratings
st.title("Recommended Books:")

# Calculate book similarities using cosine similarity
book_matrix = pd.pivot_table(ratings[ratings['user_id'].isin(top_raters)], values='rating', index='user_id', columns='book_id', fill_value=0)
similarity_matrix = cosine_similarity(book_matrix.T)

# Get the user ratings for the rated books
user_ratings_matrix = pd.pivot_table(user_ratings, values='rating', index='user_id', columns='book_id', fill_value=0)

# Calculate the weighted average of similarities and user ratings
weighted_avg = np.dot(similarity_matrix, user_ratings_matrix.values.T) / np.abs(similarity_matrix).sum(axis=1).reshape(-1, 1)

# Get the recommended book IDs
book_ids = np.argsort(weighted_avg)[-10:][::-1]

# Display the recommended books
for book_id in book_ids:
    book_title = books.loc[books['book_id'] == book_id, 'title'].values[0]
    book_author = books.loc[books['book_id'] == book_id, 'authors'].values[0]
    book_image_url = books.loc[books['book_id'] == book_id, 'image_url'].values[0]

    st.write(f"**Title:** {book_title}")
    st.write(f"**Author:** {book_author}")
    st.image(book_image_url, use_column_width=False, width=200)
