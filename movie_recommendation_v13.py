# Kelly MovieBot (Version 12)

# Import the libraries --------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction import text
import pickle
import os
import string
import decimal

# Functions used --------------------------------------------------------------------------------------------------

import warnings; warnings.simplefilter('ignore')

def get_index_from_input_movie(user_input):
    return dataset[dataset['title'].str.lower().str.replace('-', '').str.replace('the', '').str.replace(':', '').str.strip() == user_input]['index'].values[0]
    
def search_words(row, list_of_words):
    counter = 0
    for word in list_of_words:
        if word in row:
            counter = counter + 1
    return counter

def find_correct_genre(user_input, genre_list):
    scores_sim=[]
    vectorizer = TfidfVectorizer()

    for item in genre_list:
        ed = nltk.edit_distance(user_input, item)
        scores_sim.append(ed)
    correct_genre_index = scores_sim.index(min(scores_sim))
    correct_genre = genre_list[correct_genre_index].lower()
    return correct_genre

def union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list

def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)

def create_imdb_range(x):
    if x in list(drange(8, 10, '0.1')):
        return 0.2
    elif x in list(drange(6, 8, '0.1')):
        return 0.4
    elif x in list(drange(4, 6, '0.1')):
        return 0.6
    elif x in list(drange(2, 4, '0.1')):
        return 0.8
    else:
        return 1.0

def preprocess_text(raw_text):
    
    re_punc=re.compile('[%s]' % re.escape(string.punctuation))
    
    stripped=[re_punc.sub('', w) for w in raw_text.split(' ')]
    
    stripped=[token for token in stripped if token.isalpha()]
    
    #------------------------------------------------
    
    stop_words=text.ENGLISH_STOP_WORDS.union(["book"])
    
    no_stopword_text=[word for word in stripped if not word.lower() in stop_words]
    
    no_stopword_text = ' '.join(no_stopword_text) #i joined the text once more because a new lemmatizing approach is implemented below
    
    #------------------------------------------------
    
    lemmatizer = WordNetLemmatizer()
    
    #approach 1: lemmatized_text = [lemmatizer.lemmatize(word, pos='v') for word in stripped]
    #approach 1 was used until 21.02.2020, although we observed that only some of the tokens were lemmatized while others not.
    #Thus, we developed an alternative approach like below to lemmatize as many tokens/words as possible
    
    #approach 2 developed on 22.02.2020:
    lemmatized_text = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(word_tokenize(no_stopword_text))]
    
    #------------------------------------------------
    
    lowercase_text = [word.lower() for word in lemmatized_text]
    
    return ' '.join(lowercase_text)

# -----------------------------------------------------------------------------------------------


# Import the dataset

# dataset = pd.read_pickle('C:\\Users\\dq186sy\\Desktop\\Big Data Content Analytics\\Movie Recommendation System\\dataset_embedded_02092019.pkl')

dataset = pd.read_pickle(os.path.join(os.getcwd(), 'pickled_data_per_part\\dataset_part_4_29032020.pkl'))

dataset = dataset.reset_index()

dataset['index'] = np.arange(0, len(dataset))

# It is important to reset the index of the dataset in order to get the correct index per movie!

# -------------------------------------------------------------------------------------------------

def recommend_movie(input_one, input_two, input_movie):

    # Create the movieovie_genre list with the unique types of genre 

    movie_genre_list=dataset.iloc[:, 14:31].columns.tolist()

    movie_genre_list = [x.lower() for x in movie_genre_list]

    # -------------------------------------------------------------------------------------------------


    # Phase 1: Get the user's input and transform it to the appropriate form

    # First Input
    input_one = find_correct_genre(input_one.lower(), movie_genre_list)

    # Second Input
    input_movie = input_movie.lower().replace('-', '').replace('the', '').replace(':', '').strip()

    # Third Input
    input_two = input_two.lower().replace(',', '').replace('.', '')

    inputs_list=preprocess_text(input_two).split(' ')
    inputs_list = list(dict.fromkeys(inputs_list)) # remove duplicate words

    # -------------------------------------------------------------------------------------------------


    # Using the genre input given by the user, isolate those movies that match the given genre (i.e Action movies)

    lower_case_genres = []

    for i in range(len(dataset.loc[:, 'reduced_genres'])):
        lower_case_genres.append([element.lower() for element in dataset.loc[:, 'reduced_genres'].iloc[i]])
        
    dataset.loc[:,'lower_case_genres'] = lower_case_genres

    selected_rows = dataset.loc[:, 'lower_case_genres'].apply(lambda x: any(item for item in x if item == input_one))

    locked_frame = dataset[selected_rows]

    indexes_list = locked_frame.loc[:, 'index'].tolist()

    locked_frame.loc[:, 'index'] = np.arange(0, len(locked_frame))


    # -------------------------------------------------------------------------------------------------


    # Phase 2: Slice the dataset based on the user's input


    # Check of the movie user gave is in the movie list of the dataset

    selected_genre_movies_list = locked_frame['title'].str.lower().str.replace('-', '').str.replace('the', '').str.replace(':', '').str.strip().tolist()

    if input_movie in selected_genre_movies_list:

        movie_plot_new = locked_frame.loc[:, 'clean_combined_features'].loc[(locked_frame['title'].str.lower().str.replace('-', '').str.replace('the', '').str.replace(':', '').str.strip() == input_movie)].apply(lambda x: list(set(re.split(' ', x.strip().lower())))).values[0]
        
        plot_user_input_list = inputs_list + movie_plot_new

        plot_user_input_list = list(dict.fromkeys(plot_user_input_list))

        # -------------------------------------------------------------------------------------------------


        # Get the index of the movie provied by the user

        movie_index = get_index_from_input_movie(input_movie)
        
        locked_frame_index = locked_frame.loc[locked_frame['title'].str.lower().str.replace('-', '').str.replace('the', '').str.replace(':', '').str.strip() == input_movie]['index'].values[0]
        
        assert dataset.title.iloc[movie_index]==locked_frame.title.iloc[locked_frame_index]

        # -------------------------------------------------------------------------------------------------
        
        # Phase 3: Locate the word embeddings belonging to each of the three different columns (Actors, Plot, Features, Reviews)
        
        # Load the saved embeddings trained by the multi-input keras classifier
        with open(os.path.join(os.getcwd(), 'model_one\\keras_embeddings_array_concatenated_{0}_{1}_25032020.pkl'.format(str(100), str(16))), 'rb') as f:
        
            keras_embeddings_array_concatenated = pickle.load(f)

        # Phase 3.1: Locate the embeddings of the movie selected by the user!
    
        selected_movie_embeddings = keras_embeddings_array_concatenated[movie_index]
    
        selected_movie_embeddings=selected_movie_embeddings.reshape(1,-1)
        
        # Phase 3.2: Locate the embeddings of the movies that match the GENRE given by the user (i.e the embeddings of all the ACTION movies)
        
        locked_movie_embeddings = keras_embeddings_array_concatenated[indexes_list]
        
        assert selected_movie_embeddings.shape[1] == locked_movie_embeddings.shape[1]
        
        # -------------------------------------------------------------------------------------
        
        # The dimension of those two arrays should be the same.
        
        # Phase 4: Calculate Cosine Distance

        cosine_dist = cosine_distances(locked_movie_embeddings, selected_movie_embeddings.reshape(1,-1))

        # Get the similar movies & Slice the dataframe on the top 15 most similar movies to the movie given  by the user

        movie_return = np.argsort(cosine_dist, axis=None).tolist()[1:16]

        # movie_return contains the index of the 15 movies most similar to the movie selected by the user!
        
        # So the next step is to isolate those 15 movies and their features
        
        locked_frame_new = locked_frame[locked_frame.loc[:, 'index'].isin(movie_return)]

        # -------------------------------------------------------------------------------------

        # Phase 5: Create two new columns "Unique Words" + "Number of words"

        # Create the new column of "UNIQUE" words of the combined features
        locked_frame_new.loc[:, 'unique_words'] = locked_frame_new.loc[:, 'clean_combined_features']+locked_frame_new.loc[:, 'clean_reviews']

        locked_frame_new.loc[:, 'unique_words'] = locked_frame_new.loc[:, 'unique_words'].apply(lambda x: list(set(re.split(' ', x.strip().lower()))))

        locked_frame_new.loc[:, 'unique_words'] = [[x for x in lst if x] for lst in locked_frame_new.loc[:, 'unique_words']]
      
        # Create the column "Number of words" for each word contained in the unique words column

        locked_frame_new.loc[:, 'number_of_words'] = locked_frame_new.loc[:, 'unique_words'].apply(search_words, args=(plot_user_input_list,))

        # -------------------------------------------------------------------------------------
        
        # Phase 6: Recommend to the user the three most similar and highly scored movies 
    
        # Calculate the movie score
        
        locked_frame_new['imdb_rating_range']=locked_frame_new['imdb_rating'].apply(create_imdb_range)

        locked_frame_new.loc[:, 'movie_score'] = 1*locked_frame_new.loc[:, 'imdb_rating_range'].astype(float) + 0.5*locked_frame_new.loc[:, 'number_of_words'] + 0.5*locked_frame_new.loc[:, "sentiment_value"] + 0.5*locked_frame_new.loc[:, "rating"]

        # ---------------------------------------------------------------------------------------

        # Give to the user the proper movie recommendation

        top_four_rows = locked_frame_new.nlargest(4, 'movie_score')

        # Recommend the movie

        recommendations_list = top_four_rows.loc[:, ['title', 'imdb_rating', 'imdb_url']].values.tolist()
        
        return recommendations_list
        
    else:
        
        plot_user_input_list = inputs_list
        
        locked_frame.loc[:, 'unique_words'] = locked_frame.loc[:, 'clean_combined_features']+locked_frame.loc[:, 'clean_reviews']

        locked_frame.loc[:, 'unique_words'] = locked_frame.loc[:, 'unique_words'].apply(lambda x: list(set(re.split(' ', x.strip().lower()))))

        locked_frame.loc[:, 'unique_words'] = [[x for x in lst if x] for lst in locked_frame.loc[:, 'unique_words']]
      
        # Create the column "Number of words" for each word contained in the unique words column

        locked_frame.loc[:, 'number_of_words'] = locked_frame.unique_words.apply(search_words, args=(plot_user_input_list,))

        #Recommend to the user the three most similar and highly scored movies
        
        locked_frame['imdb_rating_range']=locked_frame['imdb_rating'].apply(create_imdb_range)

        locked_frame.loc[:, 'movie_score'] = 1*locked_frame.loc[:, 'imdb_rating_range'].astype(float) + 0.5*locked_frame.loc[:, 'number_of_words'] + 0.5*locked_frame.loc[:, "sentiment_value"] + 0.5*locked_frame.loc[:, "rating"]
        
        # Give to the user the proper movie recommendation

        top_four_rows = locked_frame.nlargest(4, 'movie_score')
        
        # Recommend the movie

        recommendations_list = top_four_rows.loc[:, ['title', 'imdb_rating', 'imdb_url']].values.tolist()
        
        return recommendations_list