import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
from surprise import Dataset, Reader, KNNWithMeans, SVD, NMF
from surprise.model_selection import cross_validate
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import matplotlib.pyplot as plt


# Load the small dataset
#movies_metadata = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings_small.csv')

# Select the top 20% of raters based on the number of ratings
top_raters = ratings['userId'].value_counts().head(int(len(ratings['userId'].unique()) * 1)).index
filtered_ratings = ratings[ratings['userId'].isin(top_raters)]

# Data exploration app
def data_explorer():
    st.title("Movie Data Explorer")

    # Show some basic statistics
    st.write("Number of movies:", movies_metadata.shape[0])

    # Distribution of Average Ratings
    st.header("Distribution of Average Ratings")
    average_ratings = movies_metadata['vote_average']
    average_ratings_filtered = average_ratings[average_ratings.notnull()]
    average_ratings_df = pd.DataFrame({
        'Average Rating': average_ratings_filtered
    })
    fig = alt.Chart(average_ratings_df).mark_bar().encode(
        alt.X('Average Rating', bin=alt.Bin(maxbins=50)),
        y='count()'
    )
    st.altair_chart(fig, use_container_width=True)

    # Scatterplot of the Number of Ratings per User
    st.header("Number of Ratings per User")
    user_ratings_counts = filtered_ratings['userId'].value_counts()
    scatterplot = alt.Chart(user_ratings_counts.reset_index()).mark_circle().encode(
        x='index',
        y='userId',
        size='userId',
        tooltip=['index', 'userId']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(scatterplot, use_container_width=True)

    # Top 20 Genres
    st.header("Top 50 Genres")
    genres = movies_metadata['genres'].str.split(",").explode().str.strip()
    genre_counts = genres.value_counts().head(50)
    genre_counts_df = pd.DataFrame({
        'Genre': genre_counts.index,
        'Count': genre_counts.values
    })
    bar_chart = alt.Chart(genre_counts_df).mark_bar().encode(
        x='Genre',
        y='Count'
    )
    st.altair_chart(bar_chart, use_container_width=True)


    # Movie Release Year Distribution
    st.header("Movie Release Year Distribution")
    release_years = pd.to_datetime(movies_metadata['release_date'], errors='coerce').dt.year
    release_years_filtered = release_years[release_years.notnull()]
    release_years_df = pd.DataFrame({
        'Release Year': release_years_filtered
    })
    hist = alt.Chart(release_years_df).mark_bar().encode(
        alt.X('Release Year', bin=alt.Bin(maxbins=50)),
        y='count()'
    )
    st.altair_chart(hist, use_container_width=True)


# Item-based Collaborative Filtering
#def item_based_collaborative_filtering(dataset):
#    # Compute item-item similarity matrix
#    similarity_matrix = compute_item_similarity(dataset)
#
#    # Make predictions
#    test_set = dataset.build_testset()
#    predictions = []
#    for uid, iid, true_r in test_set:
#        # Get similar items to the current item
#        similar_items = similarity_matrix[iid]
#        # Compute weighted average of ratings of similar items
#        weighted_sum = np.dot(similar_items, train_set.ur[test_set.to_inner_iid(iid)])
#        # Normalize the weighted sum by the sum of similarities
#        prediction = weighted_sum / np.sum(similar_items)
#        # Clip the prediction within the rating range
#        prediction = np.clip(prediction, train_set.rating_scale[0], train_set.rating_scale[1])
#        predictions.append((uid, iid, prediction, true_r))
#
#    # Compute RMSE
#    pred_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'est', 'true_r'])
#    rmse = accuracy.rmse(pred_df)
#
#    return pred_df

# Singular Value Decomposition (SVD)
def singular_value_decomposition(dataset):
    algo = SVD()
    results = cross_validate(algo, dataset, measures=['RMSE'], cv=5, verbose=False)
    return results

# Alternating Least Squares (ALS)
def alternating_least_squares(dataset):
    algo = NMF()
    results = cross_validate(algo, dataset, measures=['RMSE'], cv=5, verbose=False)
    return results

# Algorithm evaluator app
def algorithm_evaluator():
    st.title("Algorithm Evaluator")

    # Select the top 20% of raters based on the number of ratings
    top_raters = filtered_ratings['userId'].value_counts().head(int(len(filtered_ratings['userId'].unique()) * 0.2)).index
    filtered_ratings_top20 = filtered_ratings[filtered_ratings['userId'].isin(top_raters)]

    # Create the Surprise dataset with the filtered ratings
    surprise_dataset = Dataset.load_from_df(filtered_ratings_top20[['userId', 'movieId', 'rating']], Reader())

    # Build the train set
    train_set = surprise_dataset.build_full_trainset()

    # Perform KNNWithMeans
    st.header("KNNWithMeans")
    knn = KNNWithMeans()
    knn.fit(train_set)
    knn_predictions = knn.test(train_set.build_testset())
    knn_rmse = accuracy.rmse(knn_predictions)
    st.write("KNNWithMeans RMSE:", knn_rmse)

    # Perform NMF
    st.header("Non-negative Matrix Factorization (NMF)")
    nmf = NMF()
    nmf.fit(train_set)
    nmf_predictions = nmf.test(train_set.build_testset())
    nmf_rmse = accuracy.rmse(nmf_predictions)
    st.write("NMF RMSE:", nmf_rmse)

    # Perform item-based collaborative filtering
    #st.header("Item-based Collaborative Filtering (IBCF)")
    #item_based_results = item_based_collaborative_filtering(surprise_dataset)
    #st.write("Item-based Collaborative Filtering Results:")
    #st.write("RMSE:", np.mean(item_based_results['test_rmse']))

    # Perform Singular Value Decomposition (SVD)
    st.header("Singular Value Decomposition (SVD)")
    svd_results = singular_value_decomposition(surprise_dataset)
    st.write("SVD Results:")
    st.write("RMSE:", np.mean(svd_results['test_rmse']))

    # Perform Alternating Least Squares (ALS)
    st.header("Alternating Least Squares (ALS)")
    als_results = alternating_least_squares(surprise_dataset)
    st.write("ALS Results:")
    st.write("RMSE:", np.mean(als_results['test_rmse']))

    # Visualize algorithm performance
    algorithms = ['KNNWithMeans','Non-negative Matrix Factorization (NMF)', 'Singular Value Decomposition (SVD)', 'Alternating Least Squares (ALS)']
    rmse_values = []

    # Append the RMSE values from each algorithm's results
    rmse_values.append(knn_rmse)
    rmse_values.append(nmf_rmse)
    #rmse_values.append(np.mean(item_based_results['test_rmse']))
    rmse_values.append(np.mean(svd_results['test_rmse']))
    rmse_values.append(np.mean(als_results['test_rmse']))

    plt.bar(algorithms, rmse_values)
    plt.xlabel('Algorithm')
    plt.ylabel('RMSE')
    plt.title('Algorithm Performance')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(plt)

# Main app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", ["Data Explorer", "Algorithm Evaluator"])

    if app_mode == "Data Explorer":
        data_explorer()
    elif app_mode == "Algorithm Evaluator":
        algorithm_evaluator()

if __name__ == "__main__":
    main()

