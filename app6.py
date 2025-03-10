import pandas as pd
import numpy as np
numpy.import_array()
import sklearn.metrics.pairwise as pw
import streamlit as st

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
tags_to_include = ['comedy', 'literature', 'young-adult', 'romance', 'mystery', 'science-fiction', 'fantasy', 'horror', 'thriller', 'western', 'dystopian', 'memoir', 'biography', 'autobiography', 'history', 'travel', 'cookbook', 'self-help', 'business', 'finance', 'psychology', 'philosophy', 'religion', 'art', 'music', 'comics', 'graphic novels', 'poetry', 'sport', 'humorous', 'war', 'funny']

# Extract unique authors
unique_authors = books['authors'].unique()

# Title
st.sidebar.title("Please choose your favourite authors and/or genres")

# Allow the user to select multiple authors
selected_authors = st.sidebar.multiselect("Select authors", list(set(books['authors'].apply(lambda x: x.split(',')[0].strip()))))



# Allow the user to select multiple genres
selected_tags = st.sidebar.multiselect("Select genres", tags_to_include)

# Modify the filtered data based on the selected authors
filtered_data = book_data[book_data['authors'].isin(selected_authors) | book_data['tag_name'].isin(selected_tags)]

# Group by book and sort by count
grouped_data = filtered_data.groupby('title')['count'].sum().sort_values(ascending=False)

# Get top 5,000 raters
ratings_count = ratings.groupby('user_id').size().reset_index(name='count').sort_values('count', ascending=False)
top_raters = ratings_count[:5000]['user_id'].tolist()

# Create a DataFrame to store user ratings
user_ratings = pd.DataFrame(columns=['book_id', 'user_id', 'rating'])

# Display books to rate
st.title("Please rate these books:")
if len(grouped_data) == 0:
    st.write("No books found with selected authors or genres")
else:
    for title, count in grouped_data[:100].items():
        rating_input = st.number_input(f"Rate {title} (1-5)", min_value=1, max_value=5, key=title)
        book_id = books.loc[books['title'] == title, 'book_id'].values[0]
        user_ratings = user_ratings.append({'book_id': book_id, 'user_id': 'user1', 'rating': rating_input}, ignore_index=True)

    if st.button("Get Recommendations!"):
        # Get the ratings of the top 5,000 raters
        top_raters_ratings = ratings[ratings['user_id'].isin(top_raters)]
        top_raters_ratings = top_raters_ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

        # Add the user's ratings to the DataFrame
        user_ratings_df = pd.DataFrame(user_ratings)
        user_ratings_pivot = user_ratings_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        user_ratings_pivot = user_ratings_pivot.reindex(columns=top_raters_ratings.columns, fill_value=0)

        # Replace missing values with median
        user_ratings_pivot = user_ratings_pivot.fillna(user_ratings_pivot.median())

        # Get ratings of top 10,000 raters
        top_raters_ratings = ratings[ratings['user_id'].isin(top_raters)].pivot(index='user_id', columns='book_id', values='rating').fillna(0)

        # Merge user's ratings with top raters ratings
        merged_ratings = pd.concat([user_ratings_pivot, top_raters_ratings])

        # Calculate similarity scores using adjusted cosine similarity
        user_similarities = pw.cosine_similarity(merged_ratings, dense_output=False)[0]

        # Get the indices of the 10 closest users and their ratings
        closest_user_indices = user_similarities.argsort()[-11:-1]
        closest_user_ratings = merged_ratings.iloc[closest_user_indices]

        # Get top rated books of the 10 closest users and sort
        top_rated_books = closest_user_ratings.mean().sort_values(ascending=False)
        
        # Get recommended books, excluding those containing Potter
        user_rated_books = user_ratings_df['book_id'].tolist()
        recommended_books = []
        recommended_ids = []

        for book_id in top_rated_books.index:
            if len(recommended_books) >= 100:
                break
            title = books.loc[books['book_id'] == book_id, 'title'].values[0]
            authors = books.loc[books['book_id'] == book_id, 'authors'].values[0]
            if 'Potter' not in title and book_id not in user_rated_books:
                if title not in recommended_books:
                    recommended_books.append((title, authors))
                    recommended_ids.append(book_id)

        if len(recommended_books) == 0:
            st.write("No book recommendations found for adjusted cosine similarity.")
        else:
            st.write("Recommended books (Adjusted cosine similarity):")
            for book in recommended_books:
                st.write("- {} by {}".format(book[0], book[1]))


