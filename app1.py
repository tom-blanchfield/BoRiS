import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import requests
import csv
import base64

# Load the data
books = pd.read_csv('books.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')

# Merge tags data
book_data = pd.merge(books, book_tags, on='goodreads_book_id')
book_data = pd.merge(book_data, tags, on='tag_id')

# Define the list of genres
genre_list = ["funny", "literature", "science", "comedy", "kids", "young-adult", "romance", "mystery", "science-fiction", "fantasy", "horror",
              "thriller", "western", "chocolate", "drugs", "dystopian", "memoir", "biography", "autobiography", "history",
              "travel", "cookbook", "self-help", "business", "finance", "psychology", "philosophy", "religion",
              "art", "music", "comics", "graphic-novels", "poetry", "sport", "humorous", "war"]

# Get the list of all authors
all_authors = list(set(books['authors'].apply(lambda x: x.split(',')[0].strip())))

# Title
st.sidebar.title("Please choose whether to get your recommendations based on authors or genres, then add as many of either as you'd like, and press 'Get Recommendations!'")

# Dropdown menu to select recommendation type
selection_type = st.sidebar.selectbox("Select recommendation type", ("Authors", "Genres"))

if selection_type == "Authors":
    # Allow the user to select multiple authors to include
    selected_authors = st.sidebar.multiselect("Type authors' names to include", all_authors)
    selected_authors_exclude = st.sidebar.multiselect("Select authors to exclude", all_authors, default=[])
    
    filtered_data = book_data[book_data['authors'].apply(lambda x: x.split(',')[0].strip()).isin(selected_authors)]
    
    if len(selected_authors_exclude) > 0:
        filtered_data = filtered_data[~filtered_data['authors'].apply(lambda x: x.split(',')[0].strip()).isin(selected_authors_exclude)]
else:
    # Allow the user to select multiple genres
    selected_genres = st.sidebar.multiselect("Select genres", genre_list)
    selected_authors_exclude = st.sidebar.multiselect("Select authors to exclude", all_authors)
    
    # Create a list of book_ids by excluded authors
    excluded_author_books = set()
    for author in selected_authors_exclude:
        author_books = set(book_data[book_data['authors'].apply(lambda x: x.split(',')[0].strip()) == author]['goodreads_book_id'])
        excluded_author_books.update(author_books)

    # Filter books by selected genres
    filtered_data = book_data[book_data['tag_name'].isin(selected_genres) & ~book_data['goodreads_book_id'].isin(excluded_author_books)]

# Group by book and sort by count
grouped_data = filtered_data.groupby('tag_name').apply(lambda x: x.nlargest(21, 'count')).reset_index(drop=True)

# Create a DataFrame to store user ratings
user_ratings = pd.DataFrame(columns=['book_id', 'user_id', 'rating'])

# Display books by selected authors or genres
if selection_type == "Authors":
    st.title("Your recommendations will be generated using these books:")
else:
    st.title("Here are your genre-based recommendations:")

columns = st.columns(3)
included_books = set()
for column_idx, book in grouped_data.iterrows():
    title = book['title']
    count = book['count']
    book_id = book['book_id']
    image_url = book['image_url']

    if title in included_books:
        continue

    # Download the image from the URL
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)

        # Adjust the image size
        resized_image = image.resize((200, 300))

        # Display the book cover image, title, and author
        with columns[column_idx % 3]:
            st.image(resized_image,
                     caption=f"{title} by {books.loc[books['book_id'] == book_id, 'authors'].values[0]}",
                     use_column_width=True)

        # Add rating of 5 to user's ratings
        user_ratings = pd.concat(
            [user_ratings, pd.DataFrame({'book_id': [book_id], 'user_id': ['user_id'], 'rating': [5]})],
            ignore_index=True)

        included_books.add(title)

    except (requests.HTTPError, OSError) as e:
        st.write(f"Error loading image: {e}")

# Export CSV button for downloaded recommendations
def export_csv(data):
    filename = "recommended_books.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'Author'])
        writer.writerows(data)
    return filename

# Function to get author-based recommendations
def get_author_recommendations(selected_authors, selected_authors_exclude):
    # (Previous code for filtering and grouping by book data is not shown for brevity)

    # Get recommendations based on author ratings
    if len(selected_authors) > 0:
        st.write("Recommended books by selected authors:")

        # (Previous code for getting top raters' ratings is not shown for brevity)

        # Display recommended books
        if len(recommended_books) == 0:
            st.write("No book recommendations found.")
        else:
            st.write("Recommended books:")
            columns = st.columns(3)
            for column_idx, (title, author) in enumerate(recommended_books):
                # (Previous code for displaying book covers and creating CSV data is not shown for brevity)

            # Export CSV button for author-based recommendations
            csv_data = [(title, author) for title, author in recommended_books]
            csv_file = export_csv(csv_data)
            st.markdown(f"### [Download Recommended Books CSV](data:file/csv;base64,{base64.b64encode(open(csv_file, 'rb').read()).decode()})")

        # Genre-based recommendations
        # Filter books by selected genres
        filtered_data = book_data[book_data['tag_name'].isin(selected_genres)]

        # Group by book and sort by count
        grouped_data = filtered_data.groupby('tag_name').apply(lambda x: x.nlargest(21, 'count')).reset_index(drop=True)

        st.write("Your genre-based recommendations:")

        columns = st.columns(3)
        included_books = set()

        for column_idx, book in grouped_data.iterrows():
            title = book['title']
            author = book['authors'].split(',')[0].strip()
            book_id = book['book_id']
            image_url = book['image_url']

            # Skip if the book is already displayed or in the exclude list
            if title in included_books or author in selected_authors_exclude:
                continue

            # Download the image from the URL
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)

                # Adjust the image size
                resized_image = image.resize((200, 300))

                # Display the book cover image, title, and author
                with columns[column_idx % 3]:
                    st.image(resized_image, caption=f"{title} by {author}", use_column_width=True)

                included_books.add(title)

            except (requests.HTTPError, OSError) as e:
                st.write(f"Error loading image: {e}")

        # Export CSV button for genre-based recommendations
        csv_data = [(book['title'], book['authors'].split(',')[0].strip()) for _, book in grouped_data.iterrows()]
        csv_file = export_csv(csv_data)
        st.markdown(f"### [Download Recommended Books CSV](data:file/csv;base64,{base64.b64encode(open(csv_file, 'rb').read()).decode()})")

# Get recommendations if button is clicked
if (selection_type == "Authors" and len(selected_authors) > 0) or (selection_type == "Genres" and len(selected_genres) > 0):
    if selection_type == "Authors":
        get_author_recommendations(selected_authors, selected_authors_exclude)
    else:
        # Filter books by selected genres
        filtered_data = book_data[book_data['tag_name'].isin(selected_genres)]

        # Group by book and sort by count
        grouped_data = filtered_data.groupby('tag_name').apply(lambda x: x.nlargest(21, 'count')).reset_index(drop=True)

        st.write("Your genre-based recommendations:")

        columns = st.columns(3)
        included_books = set()

        for column_idx, book in grouped_data.iterrows():
            title = book['title']
            author = book['authors'].split(',')[0].strip()
            book_id = book['book_id']
            image_url = book['image_url']

            # Skip if the book is already displayed or in the exclude list
            if title in included_books or author in selected_authors_exclude:
                continue

            # Download the image from the URL
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)

                # Adjust the image size
                resized_image = image.resize((200, 300))

                # Display the book cover image, title, and author
                with columns[column_idx % 3]:
                    st.image(resized_image, caption=f"{title} by {author}", use_column_width=True)

                included_books.add(title)

            except (requests.HTTPError, OSError) as e:
                st.write(f"Error loading image: {e}")

        # Export CSV button for genre-based recommendations
        csv_data = [(book['title'], book['authors'].split(',')[0].strip()) for _, book in grouped_data.iterrows()]
        csv_file = export_csv(csv_data)
        st.markdown(f"### [Download Recommended Books CSV](data:file/csv;base64,{base64.b64encode(open(csv_file, 'rb').read()).decode()})")
