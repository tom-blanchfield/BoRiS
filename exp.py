import streamlit as st
import pandas as pd

# Load the data
books = pd.read_csv('books.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')

# Merge the data
book_data = pd.merge(books, book_tags, on='goodreads_book_id')
book_data = pd.merge(book_data, tags, on='tag_id')

# Get top 1000 raters
ratings_count = ratings.groupby('user_id').size().reset_index(name='count').sort_values('count', ascending=False)
top_raters = ratings_count[:1000]['user_id'].tolist()

# Get top 1000 tags
tags_to_exclude = ['to-read', 'currently-reading', 'kindle', 'audio', 'favorites']
tags_df = book_data[~book_data['tag_name'].isin(tags_to_exclude)].groupby('tag_name').size().reset_index(name='counts').sort_values('counts', ascending=False).head(1000)

# Tags to include
tags_to_include = ['favorites', 'romance', 'series', 'contemporary', 'young-adult', 'novels', 'mystery', 'adventure', 'thriller', 'humorous', 'biography', 'history', 'historical-fiction', 'graphic-novels', 'literature', 'classics']

# Get top tags that are not in the exclude list
top_tags = tags_df[tags_df['tag_name'].isin(tags_to_include)]['tag_name'].tolist()

# Sidebar with options
st.sidebar.title("Choose interests")
selected_tags = st.sidebar.multiselect("Select interests", top_tags)


# Filter the data based on the user's interests
filtered_data = book_data[book_data['tag_name'].isin(selected_tags)]

# Group by book and sort by count
grouped_data = filtered_data.groupby('title')['count'].sum().sort_values(ascending=False)

# Display recommendations
st.title("Recommended Books")
if len(grouped_data) == 0:
    st.write("No books found with selected interests")
else:
    for title, count in grouped_data[:30].items():
        rating_input = st.number_input(f"Rate {title} (1-5)", min_value=1, max_value=5, key=title)
        books.loc[books['title'] == title, 'predicted_rating'] = rating_input
        st.write(f"{title} (Count: {count})")

# Calculate Pearson correlation coefficients and recommend top 10 books
if st.button("Get Pearson"):
    # Filter ratings by top raters
    top_ratings = ratings[ratings['user_id'].isin(top_raters)]
    
    # Pivot ratings data
    ratings_pivot = top_ratings.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
    
    # Calculate Pearson correlation coefficients
    corr_matrix = ratings_pivot.corr(method='pearson', min_periods=10)
    
