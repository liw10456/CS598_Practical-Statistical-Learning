import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
@st.cache_data
def load_data():
    # Load the rating matrix and similarity matrix
    rating_matrix = pd.read_csv("I-w9Wo-HSzmUGNNHw0pCzg_bc290b0e6b3a45c19f62b1b82b1699f1_Rmat.csv", index_col=0)
    similarity_matrix = pd.read_csv("similarity_matrix.csv", index_col=0)
    return rating_matrix, similarity_matrix

# Normalize matrix
def normalize_matrix(matrix):
    row_means = matrix.mean(axis=1, skipna=True)
    return matrix.sub(row_means, axis=0)

# Recommendation function
def myIBCF(new_user_ratings, similarity_matrix):
    predictions = {}
    for movie in similarity_matrix.index:
        if pd.isna(new_user_ratings.get(movie, np.nan)):  # Only predict for unrated movies
            similar_movies = similarity_matrix[movie].dropna()
            rated_movies = new_user_ratings.dropna()
            relevant_movies = rated_movies.index.intersection(similar_movies.index)

            if len(relevant_movies) > 0:
                weights = similar_movies[relevant_movies]
                ratings = rated_movies[relevant_movies]
                denominator = weights.sum()
                numerator = (weights * ratings).sum()

                if denominator > 0:
                    predictions[movie] = numerator / denominator
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in sorted_predictions[:10]]

# Main App
def main():
    st.title("Movie Recommender System")
    st.subheader("Rate Movies to Get Personalized Recommendations")

    # Load data
    rating_matrix, similarity_matrix = load_data()

    # Show sample movies to rate
    sample_movies = rating_matrix.columns[:100]
    st.write("Please rate the following sample movies (1-5 stars or leave blank):")
    user_ratings = {}
    for movie in sample_movies:
        user_ratings[movie] = st.slider(f"{movie}", min_value=1, max_value=5, value=None, step=1, format="%d")

    # Convert user ratings to a Pandas Series
    user_ratings_series = pd.Series(user_ratings).dropna()

    # Show recommendations if ratings are provided
    if st.button("Get Recommendations"):
        if user_ratings_series.empty:
            st.warning("Please rate at least one movie to get recommendations.")
        else:
            recommendations = myIBCF(user_ratings_series, similarity_matrix)
            st.success("Top 10 Recommended Movies for You:")
            for movie in recommendations:
                st.write(movie)

if __name__ == "__main__":
    main()