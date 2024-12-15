import streamlit as st
import numpy as np
import pandas as pd

# Set up page configuration
st.set_page_config(page_title="Movie Ratings", layout="wide")

# Load the similarity matrix and movies data
@st.cache_data
def load_data():
    similarity_matrix_top30 = pd.read_csv('similarity_matrix_top30.csv', index_col=0)
    popular_movies = pd.read_csv('popular_movies.csv', index_col=0)
    movies = pd.read_csv('movies.dat', sep='::', engine='python',
                         encoding="ISO-8859-1", header=None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    multiple_idx = pd.Series([("|" in movie) for movie in movies['Genres']])
    movies.loc[multiple_idx, 'Genres'] = 'Multiple'
    rating_matrix = pd.read_csv('Rmat.csv', index_col=0, na_values=['NA'])
    rating_matrix = rating_matrix.replace('NA', pd.NA).astype(float)
    return similarity_matrix_top30, movies, popular_movies, rating_matrix


similarity_matrix_top30, movies, popular_movies, rating_matrix = load_data()

# Step 5: Implement myIBCF function
def myIBCF(newuser, similarity_matrix, top_n=10):
    newuser = pd.Series(newuser, index=similarity_matrix.columns)  # Align with matrix columns

    # Predict ratings for movies not yet rated
    predictions = {}
    for i in similarity_matrix.index:
        if pd.isna(newuser[i]):  # Only predict for unrated movies
            # Find similar movies rated by the user
            similar_movies = similarity_matrix.loc[i].dropna()
            rated_movies = newuser[~newuser.isna()]

            # Find common movies
            common_movies = similar_movies.index.intersection(rated_movies.index)

            if len(common_movies) > 0:
                sim_values = similar_movies[common_movies]
                weighted_sum = np.sum(sim_values * rated_movies[common_movies])
                normalization = np.sum(sim_values)

                if normalization > 0:
                    predictions[i] = weighted_sum / normalization

    # Sort predictions
    predicted_ratings = pd.Series(predictions).sort_values(ascending=False)

    # Select the top_n recommendations
    top_recommendations = predicted_ratings.head(top_n)

    # Filter out NA values from top_recommendations
    non_na_recommendations = top_recommendations[~top_recommendations.isna()]

    if len(non_na_recommendations) < top_n:
        # Make a copy of popular_movies and modify MovieID to have a prefix 'm'
        popular_movies_with_m = popular_movies.copy()
        print("Non-NA Recommendations are < 10")
        popular_movies_with_m['MovieID'] = 'm' + popular_movies_with_m['MovieID'].astype(str)
        popular_movies_with_m.set_index('MovieID', inplace=True)

        # Get movies already rated by the user
        already_rated = newuser[~newuser.isna()].index  # Movies already rated by the user

        # Ensure indices are consistent (convert to strings if needed)
        already_rated = already_rated.astype(str)
        non_na_recommendations.index = non_na_recommendations.index.astype(str)
        popular_movies_with_m.index = popular_movies_with_m.index.astype(str)

        # Identify remaining popular movies that are neither recommended nor already rated
        remaining_popularity = popular_movies_with_m.index.difference(
            non_na_recommendations.index.union(already_rated)
        )

        # Debug: Check which movies are being excluded
        print("Already Rated Movies:", already_rated)
        print("Recommended Movies:", non_na_recommendations.index)
        print("Remaining Popular Movies:", remaining_popularity)

        remaining_popularity_sorted = popular_movies_with_m.loc[remaining_popularity].sort_values(by='AvgRating',
                                                                                                  ascending=False).index
        print("Remaining Popular Movies Sorted:", remaining_popularity_sorted)
        # Get the top remaining popular movies to fill the gap
        fallback = popular_movies_with_m.loc[remaining_popularity_sorted].head(top_n - len(non_na_recommendations))

        # Debug: Check the fallback movies
        print("Fallback Movies:")
        print(fallback)
        # Add fallback recommendations to non-NA recommendations
        top_recommendations = pd.concat([non_na_recommendations, fallback["AvgRating"]])

        # Ensure no overlap between recommendations and already rated movies
        top_recommendations = top_recommendations[~top_recommendations.index.isin(already_rated)]

        print("Top Recommendations (including fallbacks):")
        print(top_recommendations)

        # Debugging: Check for overlap between recommendations and already rated movies
        overlap = top_recommendations.index.intersection(already_rated)
        print("User has already rated the following recommended movies:", overlap)
        print((len(overlap) == 0, overlap))  # Expect this to be (True, Index([]))

        print(f"Recommendations for User - :\n", top_recommendations)

    # Ensure we return only the top_n recommendations
    # top_recommendations = top_recommendations.head(top_n)
    # Return top_n recommendations
    return top_recommendations


# Title of the application
#st.title("Movie Recommendation System")
st.markdown("<h2 style='text-align: center; color: #4a4a4a;'>PSL [CS598] - Movie Recommendation System</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #4a4a4a;'>Anikesh Haran</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #4a4a4a;'>anikesh2@illinois.edu</h5>", unsafe_allow_html=True)
# Limit to top 500 movies
top_movies = popular_movies.head(500)

# URL template for movie posters
POSTER_URL_TEMPLATE = "https://liangfgithub.github.io/MovieImages/{}.jpg"

# Page title
#st.markdown("<h4 style='text-align: center; color: #4a4a4a;'>Rate Popular Movies</h1>", unsafe_allow_html=True)

# Layout for tiles with row-spanning
cols_per_row = 4  # Number of tiles per row
num_movies = len(top_movies)

if 'expander_expanded' not in st.session_state:
    st.session_state.expander_expanded = True

# Dynamically calculate the row span
with st.expander("Rate Popular Movies", expanded=st.session_state.expander_expanded):
    with st.container(height=500):
        # Initialize ratings dictionary in session state if it doesn't exist
        if 'ratings' not in st.session_state:
            st.session_state.ratings = {}

        rows = (num_movies + cols_per_row - 1) // cols_per_row
        # Display movies in tiles
        for i in range(rows):
            # Dynamically create columns
            cols = st.columns(cols_per_row)

            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_movies:
                    row = top_movies.iloc[idx]
                    movie_id = row["MovieID"]
                    title = row["Title"]
                    poster_url = POSTER_URL_TEMPLATE.format(movie_id)

                    with cols[j]:
                        # Display movie poster
                        if len(title) > 20:
                            short_title = title[:20] + "..."
                        else:
                            short_title = title
                        st.image(poster_url, caption=short_title, width=200)

                        # Display rating slider
                        rating = st.slider(f"Rate", min_value=0, max_value=5, step=1, key=f"rating_{movie_id}", label_visibility="hidden")

                        # Store the rating
                        # Convert NumPy integer to Python integer
                        movie_id = int(movie_id)
                        st.session_state.ratings[movie_id] = rating

# Button to submit ratings
with st.expander("Recommendations for You", expanded=True):
    if st.button("Submit Ratings"):
        # Collapse the expander
        st.session_state.expander_expanded = False
        # Filter out movies with no ratings
        rated_movies = {movie_id: rating for movie_id, rating in st.session_state.ratings.items() if rating > 0}

        # Display the rated movies
        #st.write("### Your Ratings:")
        #st.write(rated_movies)
        # Create a hypothetical user vector
        hypothetical_user = np.full(rating_matrix.shape[1], np.nan)
        for movie_id, rating in rated_movies.items():
            movie_index = movies[movies['MovieID'] == movie_id].index[0]
            hypothetical_user[movie_index] = rating

        # Compute recommendations for the hypothetical user
        recommendations_hypothetical = myIBCF(hypothetical_user, similarity_matrix_top30)

        # Display recommendations
        #st.write("### Recommendations for You:")
        #st.write(recommendations_hypothetical)
        # Display recommendations container
        with st.container():
            st.write("### Recommendations for You:")

            # Define number of columns per row (adjust as needed)
            cols_per_row = 4
            num_movies = len(recommendations_hypothetical)

            # Calculate number of rows
            rows = (num_movies + cols_per_row - 1) // cols_per_row

            # Display movies in tiles
            for i in range(rows):
                # Dynamically create columns
                cols = st.columns(cols_per_row)

                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < num_movies:
                        movie_id = recommendations_hypothetical.index[idx]
                        # Remove 'm' prefix from movie ID
                        movie_id = str(movie_id)[1:]
                        print(movie_id)
                        # Get movie details from movies DataFrame
                        row = movies[movies['MovieID'] == int(movie_id)]
                        print(row)
                        movie_details = row.values[0]
                        title = movie_details[1]
                        genres = movie_details[2]
                        poster_url = POSTER_URL_TEMPLATE.format(movie_id)
                        with cols[j]:
                            # Display movie poster
                            st.image(poster_url, caption=title, width=200)
