from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import *
import ydata_profiling as pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import pickle
import requests
import cv2
# import matplotlib.pyplot as plt
from deepface import DeepFace
import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.utils import load_img,img_to_array
from keras.models import  load_model

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import random


from bs4 import BeautifulSoup as SOUP
import re
import requests as HTTP

# Main Function for scraping


def get_movie_titles_by_emotion(emotion, csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Mapping emotions to genre IDs
    emotion_to_genre = {
        'sad':35,
        'disgust':12,
        'anger':53,
        'happy':28,
        'neutral':878,
        'surprise':9648,
        'fear':16
    }

    # Check if the provided emotion is in the emotion_to_genre dictionary
    if emotion in emotion_to_genre:
        genre_id = emotion_to_genre[emotion]

        # Filter the DataFrame based on the genre ID
        print(genre_id)
        title_list=[]
        for i,k in enumerate (df['genres']):
            # print(df['title'][i])
            for j in eval(k):
                # print(j)
                if j['id'] == genre_id:
                    title_list.append(df['title'][i])

                    # print(df[str(i)])

        if title_list !=[]:
            random_titles = random.sample(title_list, 4)
            return random_titles
        else:
            return "No movies found for the given emotion and genre."

    else:
        return "No genre mapping found for the given emotion."










CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

model = load_model("./Emotion_Detection/best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend_music(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        artist = music.iloc[i[0]].artist
        print(artist)
        print(music.iloc[i[0]].song)
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
        recommended_music_names.append(music.iloc[i[0]].song)

    return recommended_music_names,recommended_music_posters



def fetch_poster(movie_id):
     response= requests.get(
'https://api.themoviedb.org/3/movie/{}?api_key=7361490d77f5f70da4ec1467f6e1d4ee&language=en-US'.format(movie_id))
     data=response.json()
     return "https://image.tmdb.org/t/p/w500" +data['poster_path']

def recommend(movie):
     movie_index = movies[movies['title'] == movie].index[0]
     distances = similarity[movie_index]
     movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

     recomended_movies=[]
     recomended_movies_poster=[]
     for i in movie_list:
          movie_id=movies.iloc[i[0]].id
          recomended_movies.append(movies.iloc[i[0]].title)
          recomended_movies_poster.append(fetch_poster(movie_id))
     return recomended_movies,recomended_movies_poster


movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies =pd.DataFrame(movies_dict)
print(movies)
similarity= pickle.load(open('similarity.pkl','rb'))


if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice2 = st.radio("Select an Option", ["Recommendation System"])

    # Depending on the selected option, you can show specific content
    if choice2 == "Data Visualization":
        choice = st.radio("Choose a Data Visualization option",
                                   ["Upload", "Profiling", "Modelling"])

        st.info("This project application helps you build and explore your data.")
        # Handle actions based on the selected data visualization option

    elif choice2 == "Recommendation System":
        choice = st.radio("Choose a Recommendation System option", ["Movie & music","Food"])
        st.info("This project application helps you Get Movie and Music reccomendation by learning your mood Through facial expression.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)

    # Handle missing values in the target column
    df = df.dropna(subset=[chosen_target])

    st.dataframe(df)  # Display the DataFrame after handling missing values

    if st.button('Run Modelling'):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)

        best_model = compare_models()

        compare_df = pull()

        # Display all accuracy scores
        st.subheader('All Accuracy Scores')
        st.dataframe(compare_df)

        # Display the best model
        st.subheader('Best Model')
        st.write(best_model)


        st.subheader('Why this Model?')
        st.write(
            "The best model was chosen based on the highest performance metrics, including R2, MAE, MSE, RMSE, and MAPE. "
            "These metrics collectively indicate the model's ability to accurately predict the target variable.")

        save_model(best_model, 'best_model')

        st.subheader('Download the Model')
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

if choice=="Movie & music":
    st.title('Movie & Music Recommender System')

    tab1,tab2 = st.tabs(["capture","recommend"])

    with tab1:
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            st.header("The shape of the Image is:-")
            st.write(cv2_img.shape)

            result = DeepFace.analyze(cv2_img,
                                      actions=['emotion'])

            # print result
            # st.write(len(result))
            st.write("Predicted Expression:- " + result[0]["dominant_emotion"])
            emotion = result[0]["dominant_emotion"]
            emotion_input = emotion
            csv_file_path = "./tmdb_5000_movies.csv"
            result = get_movie_titles_by_emotion(emotion_input, csv_file_path)
            # print(result)
            # a = main(emotion)
            st.write(" ")
            st.header("Suggested Movies")
            for i,j in enumerate(result):
                st.write(str(i+1)+") "+j)


    with tab2:
        tab3, tab4 = st.tabs(["Movie", "Music"])
        with tab3:
            st.header("Movie Recommender System")
            selected_movie_name = st.selectbox(
                'How would you like to be contacted?',
                (movies['title'].values))
            if st.button('Recommend'):
                names, poster = recommend(selected_movie_name)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.text(names[0])
                    st.image(poster[0])
                with col2:
                    st.text(names[1])
                    st.image(poster[1])
                with col3:
                    st.text(names[2])
                    st.image(poster[2])
                with col4:
                    st.text(names[3])
                    st.image(poster[3])
                with col5:
                    st.text(names[4])
                    st.image(poster[4])
        with tab4:
            st.header('Music Recommender System')
            music = pickle.load(open('df.pkl', 'rb'))
            similarity = pickle.load(open('similarity.pkl', 'rb'))

            music_list = music['song'].values
            selected_movie = st.selectbox(
                "Type or select a song from the dropdown",
                music_list
            )

            if st.button('Show Recommendation'):
                recommended_music_names, recommended_music_posters = recommend_music(selected_movie)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.text(recommended_music_names[0])
                    st.image(recommended_music_posters[0])
                with col2:
                    st.text(recommended_music_names[1])
                    st.image(recommended_music_posters[1])

                with col3:
                    st.text(recommended_music_names[2])
                    st.image(recommended_music_posters[2])
                with col4:
                    st.text(recommended_music_names[3])
                    st.image(recommended_music_posters[3])
                with col5:
                    st.text(recommended_music_names[4])
                    st.image(recommended_music_posters[4])

if choice == "Food":

    st.title("Food Recommendation System")
    st.text("Let us help you with ordering")
    st.image("foood.jpeg")

    ## nav = st.sidebar.radio("Navigation",["Home","IF Necessary 1","If Necessary 2"])

    st.subheader("Whats your preference?")
    vegn = st.radio("Vegetables or none!", ["veg", "non-veg"], index=1)

    st.subheader("What Cuisine do you prefer?")
    cuisine = st.selectbox("Choose your favourite!",
                           ['Healthy Food', 'Snack', 'Dessert', 'Japanese', 'Indian', 'French',
                            'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai'])

    st.subheader("How well do you want the dish to be?")  # RATING
    val = st.slider("from poor to the best!", 0, 10)

    food = pd.read_csv(r".\FRS\input\food.csv")
    ratings = pd.read_csv(r".\FRS\input\ratings.csv")
    combined = pd.merge(ratings, food, on='Food_ID')
    # ans = food.loc[(food.C_Type == cuisine) & (food.Veg_Non == vegn),['Name','C_Type','Veg_Non']]

    ans = combined.loc[
        (combined.C_Type == cuisine) & (combined.Veg_Non == vegn) & (combined.Rating >= val), ['Name', 'C_Type',
                                                                                               'Veg_Non']]
    names = ans['Name'].tolist()
    x = np.array(names)
    ans1 = np.unique(x)

    finallist = ""
    bruh = st.checkbox("Choose your Dish")
    if bruh == True:
        finallist = st.selectbox("Our Choices", ans1)

    ##### IMPLEMENTING RECOMMENDER ######
    dataset = ratings.pivot_table(index='Food_ID', columns='User_ID', values='Rating')
    dataset.fillna(0, inplace=True)
    csr_dataset = csr_matrix(dataset.values)
    dataset.reset_index(inplace=True)

    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model.fit(csr_dataset)


    # st.write(food)
    def food_recommendation(Food_Name):
        n = 10
        FoodList = food[food['Name'].str.contains(Food_Name)]
        if len(FoodList):
            Foodi = FoodList.iloc[0]['Food_ID']
            Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
            distances, indices = model.kneighbors(csr_dataset[Foodi], n_neighbors=n + 1)
            Food_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                  key=lambda x: x[1])[:0:-1]
            Recommendations = []
            for val in Food_indices:
                Foodi = dataset.iloc[val[0]]['Food_ID']
                i = food[food['Food_ID'] == Foodi].index
                Recommendations.append({'Name': food.iloc[i]['Name'].values[0], 'Distance': val[1]})
            df = pd.DataFrame(Recommendations, index=range(1, n + 1))
            return df['Name']
        else:
            return "No Similar Foods."


    display = food_recommendation(finallist)
    # names1 = display['Name'].tolist()

    # x1 = np.array(names)
    # ans2 = np.unique(x1)
    if bruh == True:
        bruh1 = st.checkbox("We also Recommend : ")
        if bruh1 == True:
            for i in display:
                st.write(i)
    # recommend_food()
