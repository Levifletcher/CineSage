import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Vectorize genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title'])

# Recommender function
def recommend(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Streamlit UI
st.title("🧙🏻‍♂️ Cinesage")
movie_list = movies['title'].values
selected_movie = st.selectbox("Pick a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("**Recommended Movies:**")
    for movie in recommendations:
        st.write(movie)
