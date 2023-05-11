import pandas as pd
import streamlit as st

# Load the movie dataset
movie_data = pd.read_csv("imdb_top_1000.csv")

# Load the recommendations dataset
recommendations = pd.read_csv("recommendations_bert.csv")

# Set the page title and favicon
st.set_page_config(page_title="WebUI", page_icon=":clapper:")

# Set the title of the web app
st.title("Movie Recommendations")

# Create a text input box to enter the movie name
selected_movie = st.text_input("Enter the title of the movie")

# Check if user has entered any text
if selected_movie:

    # Find the recommendations for the selected movie
    recommendation_row = recommendations.loc[recommendations["Watched Movie"].str.lower() == selected_movie.lower()]

    # If the movie is not found in the recommendations file, display an error message
    if recommendation_row.empty:
        st.write("Movie not found in recommendations.")

    # If the movie is found, display its top 5 recommendations
    else:
        st.write(f"Top 5 recommendations for {selected_movie}:")
        for i in range(1, 6):
            recommended_movie = recommendation_row.iloc[0, i]
            recommended_movie_data = movie_data.loc[movie_data["Series_Title"].str.lower() == recommended_movie.lower()]
            if not recommended_movie_data.empty:
                poster = recommended_movie_data["Poster_Link"].values[0]
                title = recommended_movie_data["Series_Title"].values[0]
                year = recommended_movie_data["Released_Year"].values[0]
                st.image(poster, caption=f"{title} ({year})", width=100)
            else:
                st.write(f"{recommended_movie} not found in movie data.")
