import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity as cos_sim 
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
books = pd.read_csv('books.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')

# Set page title
st.set_page_config(page_title="Book Explorer")

# Create a sidebar with options
st.sidebar.title("Explore the Book Data")
option = st.sidebar.selectbox("Select an option", ("Books", "Book Tags", "Ratings", "Get cosine similarities of top 100 raters", "Get cosine similarities of top 1,000 raters" ))

if option == "Books":
    #Show the Books dataframe
    st.header("Books Data")
    st.dataframe(books)
    
#if option == "Book Tags":
#    # Add a chart to show the top 50 most popular tags
#    top_tags = book_tags.groupby('tag_id').count().sort_values('count', ascending=False).head(50)
#    top_tags.reset_index(inplace=True)
#    top_tags['tag_name'] = top_tags['tag_id'].apply(lambda x: tags.loc[tags['tag_id']==x, 'tag_name'].values[0])
#    chart_data = top_tags[['tag_name', 'count']]
#    bars = alt.Chart(chart_data).mark_bar().encode(
#        x='count',
#        y=alt.Y('tag_name', sort='-x'),
#        tooltip=['tag_name', 'count']
#    ).properties(
#        title='Top 50 Most Popular Book Tags'
#    )
#    st.altair_chart(bars, use_container_width=True)
if option == "Book Tags":
    # Generate the list of genres
    genre_list = ["comedy", "literature", "irish", "superheroes", "science", "young-adult", "science-fiction", "romance", "mystery", "fantasy", "horror",
                  "paranormal", "thriller", "western", "dystopian", "memoir", "biography", "autobiography", "history",
                  "travel", "cookbook", "self-help", "business", "finance", "war", "psychology", "philosophy", "religion",
                  "art", "music", "comics", "graphic-novels", "poetry", "football", "sport", "funny"]
    
    # Add a chart to show the top 50 most popular tags
    top_tags = book_tags.groupby('tag_id').count().sort_values('count', ascending=False).head(50)
    top_tags.reset_index(inplace=True)
    top_tags['tag_name'] = top_tags['tag_id'].apply(lambda x: tags.loc[tags['tag_id']==x, 'tag_name'].values[0])
    chart_data = top_tags[['tag_name', 'count']]
    tags_bars = alt.Chart(chart_data).mark_bar().encode(
        x='count',
        y=alt.Y('tag_name', sort='-x'),
        tooltip=['tag_name', 'count']
    ).properties(
        title='Top 50 Most Popular Book Tags'
    )
    
    # Display the bar chart for genres
    genre_counts = pd.Series(genre_list).value_counts()
    genre_chart_data = pd.DataFrame({'Genre': genre_counts.index, 'Count': genre_counts.values})
    
    # Set a maximum of 5000 for the x-axis
    max_count = 5000
    genre_bars = alt.Chart(genre_chart_data).mark_bar().encode(
        x=alt.X('Count', scale=alt.Scale(domain=(0, max_count))),
        y=alt.Y('Genre', sort='-x'),
        tooltip=['Genre', 'Count']
    ).properties(
        title='Distribution of Genres'
    )
    
    # Display the charts
    st.altair_chart(tags_bars, use_container_width=True)
    st.altair_chart(genre_bars, use_container_width=True)
    
    # Display the charts
    st.altair_chart(tags_bars, use_container_width=True)
    st.altair_chart(genre_bars, use_container_width=True)        
    
if option == "Ratings":
    # Add a chart to show the distribution of average ratings
    chart_data = books[['average_rating']]
    hist = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("average_rating", bin=alt.Bin(maxbins=50)),
        y='count()',
        tooltip='count()'
    ).properties(
        title='Distribution of Average Ratings'
    )
    st.altair_chart(hist, use_container_width=True)
    
    
    # Add a chart to show the distribution of ratings
    chart_data = ratings['rating'].value_counts().sort_index().reset_index()
    chart_data.columns = ['Rating', 'Count']
    pie_chart = alt.Chart(chart_data).mark_arc().encode(
        theta='Count',
        color='Rating',
        tooltip=['Rating', 'Count']
    ).properties(
        title='Distribution of Ratings'
    )
    st.altair_chart(pie_chart, use_container_width=True)

    # Add a scatterplot to show the number of times raters have rated
    rating_counts = ratings['user_id'].value_counts()
    scatter_data = pd.DataFrame({'User': rating_counts.index, 'Rating Count': rating_counts.values})
    scatterplot = alt.Chart(scatter_data).mark_circle().encode(
        x='User',
        y='Rating Count',
        tooltip=['User', 'Rating Count']
    ).properties(
        title='Number of Times Raters Have Rated'
    )
    st.altair_chart(scatterplot, use_container_width=True)
    
    #This chart will only display if expanded but it's worth it!
    
    # Add a chart to show the number of times raters have rated
    #st.title ("This chart is hidden on load, click the expand button ->")
    #chart_data = ratings['user_id'].value_counts().reset_index()
    #chart_data.columns = ['user_id', 'rating_count']
    #bars = alt.Chart(chart_data).mark_bar().encode(
    #    x='rating_count',
    #    y=alt.Y('user_id:O', sort='-x'),
    #    tooltip=['user_id', 'rating_count']
    #).properties(
    #    title='Number of Times Raters Have Rated'
    #)
    #st.altair_chart(bars, use_container_width=True)
    
    #Show the ratings dataframe
    #st.header("Ratings Data")
    #st.dataframe(ratings)  
    
    # Add a button to generate cosine similarities for 100 top raters
if option == "Get cosine similarities of top 100 raters":
    # Find top 1000 users by ratings count
    top_users = ratings['user_id'].value_counts().index[:100]
    # Filter ratings by top users
    ratings_top = ratings[ratings['user_id'].isin(top_users)]
    # Convert ratings to a sparse matrix
    user_book_ratings = pd.pivot_table(ratings_top, values='rating', index=['user_id'], columns=['book_id'])
    user_book_ratings = user_book_ratings.fillna(0)
    user_book_ratings_sparse = sk.preprocessing.normalize(user_book_ratings, axis=0)
    # Compute cosine similarity between users
    sim = cos_sim(user_book_ratings_sparse)
    # Convert cosine similarity to DataFrame
    sim_df = pd.DataFrame(sim)
    # Plot heatmap
    sns.heatmap(sim_df, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title("Cosine Similarity Heatmap of top 100 raters")
    st.pyplot()

    # Add a button to generate cosine similarities for 1,000 top raters
if option == "Get cosine similarities of top 1,000 raters":
    top_users = ratings['user_id'].value_counts().index[:1000]
    # Filter ratings by top users
    ratings_top = ratings[ratings['user_id'].isin(top_users)]
    # Convert ratings to a sparse matrix
    user_book_ratings = pd.pivot_table(ratings_top, values='rating', index=['user_id'], columns=['book_id'])
    user_book_ratings = user_book_ratings.fillna(0)
    user_book_ratings_sparse = sk.preprocessing.normalize(user_book_ratings, axis=0)
    # Compute cosine similarity between users
    sim = cos_sim(user_book_ratings_sparse)
    # Convert cosine similarity to DataFrame
    sim_df = pd.DataFrame(sim)
    # Plot heatmap
    sns.heatmap(sim_df, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title("Cosine Similarity Heatmap of top 1,000 raters")
    st.pyplot()
    




  


    
