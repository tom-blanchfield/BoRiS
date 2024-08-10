import streamlit as st
import pandas as pd

# Load the data
books = pd.read_csv('books.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')

# Merge the data
book_data = pd.merge(books, book_tags, on='goodreads_book_id')
book_data = pd.merge(book_data, tags, on='tag_id')

# Get top 20 tags
tags_to_exclude = ['to-read', 'currently-reading', 'kindle', 'audio', 'favorites']
tags_to_include = ['favorites', 'romance', 'series', 'contemporary', 'young-adult', 'novels', 'mystery', 'adventure', 'thriller',]
top_tags = book_data[~book_data['tag_name'].isin(tags_to_exclude)][book_data['tag_name'].isin(tags_to_include)].groupby('tag_name').size().reset_index(name='counts').sort_values('counts', ascending=False).head(20)['tag_name'].tolist()

# Sidebar with options
st.sidebar.title("Choose interests")
selected_tags = []
for tag in top_tags:
    if st.sidebar.checkbox(tag):
        selected_tags.append(tag)

# Show top 20 tags
st.sidebar.title("Top 20 tags")
tags_df = book_data[~book_data['tag_name'].isin(tags_to_exclude)][book_data['tag_name'].isin(tags_to_include)].groupby('tag_name').size().reset_index(name='counts').sort_values('counts', ascending=False).head(20)
st.sidebar.write(tags_df)

# Filter the data based on the user's interests
filtered_data = book_data[book_data['tag_name'].isin(selected_tags)]

# Group by book and sort by count
grouped_data = filtered_data.groupby('title')['count'].sum().sort_values(ascending=False)

# Display recommendations
st.title("Recommended Books")
if len(grouped_data) == 0:
    st.write("No books found with selected interests")
else:
    for title, count in grouped_data[:10].items():
        st.write(f"{title} (Count: {count})")
