#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
from streamlit_star_rating import st_star_rating


# In[ ]:


st.set_page_config(page_title="Movie Ratings", layout="wide")
cols_per_row = 5  # Number of tiles per row


# Import the similarity matrix and movie dataset.

# In[ ]:


@st.cache_data
def load_data():
    # Load similarity matrix
    similarity_matrix = pd.read_csv('similarity_matrix.csv', index_col=0)
    
    # Load popular movies
    popular_movies = pd.read_csv('popular_movies.csv', index_col=0)
    
    # Load and preprocess movies data
    movies = pd.read_csv('movies.dat', sep='::', engine='python',
                         encoding="ISO-8859-1", header=None, names=['MovieID', 'Title', 'Genres'])
    # Simplify genres for movies with multiple genres
    movies['Genres'] = movies['Genres'].apply(lambda genre: 'Multiple' if '|' in genre else genre)
    
    # Load and preprocess rating matrix
    rating_matrix = pd.read_csv('Rmat.csv', index_col=0, na_values=['NA'])
    rating_matrix = rating_matrix.astype(float)
    
    return similarity_matrix, movies, popular_movies, rating_matrix


# In[ ]:


similarity_matrix, movies, popular_movies, rating_matrix = load_data()


# Step 5: Implement myIBCF function

# In[ ]:


def myIBCF(newuser, similarity_matrix, top_n=10):
    # Align the user's ratings with similarity matrix columns
    newuser = pd.Series(newuser, index=similarity_matrix.columns)

    # Identify unrated movies
    unrated_movies = newuser.isna()

    # Filter the similarity matrix for movies the user has rated
    rated_movies = newuser.dropna()
    similar_movies_matrix = similarity_matrix.loc[rated_movies.index]  # Similarity matrix for rated movies

    # Weighted sum and normalization (vectorized)
    weighted_sums = np.dot(similar_movies_matrix.T, rated_movies)  # Dot product

    # Calculate normalization factor (sum of valid similarities for each movie)
    normalizations = similarity_matrix.notna().sum(axis=1)  # Sum of valid similarities per movie

    # Ensure normalizations is a vector with the same length as weighted_sums
    if weighted_sums.shape != normalizations.shape:
        raise ValueError(f"Shapes of weighted_sums ({weighted_sums.shape}) and normalizations ({normalizations.shape}) do not match")

    # Calculate predictions
    predictions = weighted_sums / normalizations
    predictions = predictions[unrated_movies]  # Filter only unrated movies

    # Sort the predictions
    predicted_ratings = predictions.sort_values(ascending=False)

    # Select top_n recommendations
    top_recommendations = predicted_ratings.head(top_n)

    # If there are fewer than `top_n` recommendations, fill the gap with popular movies
    if len(top_recommendations) < top_n:
        print("Non-NA Recommendations are < 10")

        # Get movies already rated by the user
        already_rated = newuser.dropna().index.astype(str)

        # Prepare the movie dataset for fallback
        popular_movies_with_m = popular_movies.copy()
        popular_movies_with_m['MovieID'] = 'm' + popular_movies_with_m['MovieID'].astype(str)
        popular_movies_with_m.set_index('MovieID', inplace=True)

        # Find movies that are not recommended or already rated
        remaining_popularity = popular_movies_with_m.index.difference(
            top_recommendations.index.union(already_rated.astype(str))
        )

        # Sort remaining popular movies based on their average rating
        remaining_popularity_sorted = popular_movies_with_m.loc[remaining_popularity].sort_values(by='AvgRating', ascending=False)

        # Get the top remaining popular movies to fill the gap
        fallback = remaining_popularity_sorted.head(top_n - len(top_recommendations))

        # Add fallback recommendations to top recommendations
        top_recommendations = pd.concat([top_recommendations, fallback["AvgRating"]])

        # Remove already rated movies from the final recommendations
        top_recommendations = top_recommendations[~top_recommendations.index.isin(already_rated)]

    # Ensure only top_n recommendations are returned
    return top_recommendations.head(top_n)


# <!-- Title of the application<br>
# t.title("Movie Recommendation System") -->

# In[ ]:


# Define custom button styles using Streamlit's markdown capabilities
button_style = """
<style>
div.stButton > button {
    background-color: #4CAF50; /* Green background */
    color: white;
    border: none;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    cursor: pointer;
    border-radius: 5px;
    position: relative;
    margin-left:45%
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

top_movies = popular_movies.head(200)


# Layout for tiles with row-spanning

# In[ ]:


num_movies = len(top_movies)


# In[ ]:


if 'expander_expanded' not in st.session_state:
    st.session_state.expander_expanded = True


# In[ ]:


POSTER_URL_TEMPLATE = "https://liangfgithub.github.io/MovieImages/{}.jpg"


# Dynamically calculate the row span

# In[ ]:


with st.expander("", expanded=st.session_state.expander_expanded):
    st.html(
    "<h3 style='text-align:center;'>Step 1: Give Your Ratings</h3>")
    with st.container(height=800):
        # 初始化 ratings 字典
        if 'ratings' not in st.session_state:
            st.session_state.ratings = {}

        rows = (num_movies + cols_per_row - 1) // cols_per_row  # 計算需要的行數
        
        # 顯示電影卡片
        for i in range(rows):
            # 動態創建列
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i * cols_per_row + j
                if idx < num_movies:
                    # 取得當前電影資料
                    row = top_movies.iloc[idx]
                    movie_id = int(row["MovieID"])  # 確保是 Python 整數
                    title = row["Title"]
                    poster_url = POSTER_URL_TEMPLATE.format(movie_id)

                    # 顯示電影資訊
                    with col:
                        short_title = title[:20] + "..." if len(title) > 20 else title
                        st.image(poster_url, caption=short_title, width=200)

                        # 星級評分
                        star = st_star_rating(
                            label="Please rate your experience",
                            maxValue=5,
                            defaultValue=0,
                            size=25,
                            key=f"rating_{movie_id}",
                            customCSS=("""
                                div#root {background-color:#e6fffb!important;}
                                div#root > h3 {font-size:16px!important;} 
                                div#root > div {padding: 0% 0% 0% 9%;
                            """),
                            emoticons=True
                        )
                        # 保存評分結果
                        st.session_state.ratings[movie_id] = star


# Step 2: submit

# In[ ]:


# Custom CSS to style the Streamlit UI
st.markdown(
    '''
    <style>
    .stExpander, div#root {
        background-color: #e6fffb;
    }
    div#root {
        background-color: #e6fffb !important;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# Expander section with centered step instructions
with st.expander("", expanded=True):
    st.markdown(
        "<h3 style='text-align: center;'>Step 2: Generate Recommendation by clicking on the submit button below</h3>",
        unsafe_allow_html=True,
    )
    if st.button("Submit"):
        # Filter out movies with no ratings
        rated_movies = {
            movie_id: star
            for movie_id, star in st.session_state.ratings.items()
            if star > 0
        }

        # Create a hypothetical user vector
        hypothetical_user = np.full(rating_matrix.shape[1], np.nan)
        for movie_id, star in rated_movies.items():
            movie_index = movies[movies['MovieID'] == movie_id].index[0]
            hypothetical_user[movie_index] = star

        # Compute recommendations for the hypothetical user
        recommendations_hypothetical = myIBCF(hypothetical_user, similarity_matrix)

        # Display recommendations
        with st.container():
            num_movies = len(recommendations_hypothetical)
            rows = (num_movies + cols_per_row - 1) // cols_per_row  # Calculate rows required

            # Render movie recommendations in a grid
            for i in range(rows):
                cols = st.columns(cols_per_row)  # Dynamically create columns
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < num_movies:
                        # Extract movie details
                        movie_id = recommendations_hypothetical.index[idx]
                        movie_id = str(movie_id)[1:]  # Remove 'm' prefix
                        row = movies[movies['MovieID'] == int(movie_id)]
                        if not row.empty:
                            title, genres = row.iloc[0][['Title', 'Genres']]
                            poster_url = POSTER_URL_TEMPLATE.format(movie_id)

                            with cols[j]:
                                # Display movie title, poster, and genres
                                st.markdown(f'<h5>#{idx + 1} choice</h5>', unsafe_allow_html=True)
                                st.image(poster_url, caption=title, width=200)
                                st.html("<span style='padding-left:25%'>"f'{genres}'"</span>")

