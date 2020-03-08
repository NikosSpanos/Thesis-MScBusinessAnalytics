# Kelly MovieBot (Version 12)

# Import the libraries --------------------------------------------------------------------------------------------------

import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

import requests

import nltk

from nltk import word_tokenize

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_distances

import pickle

import os

# Functions used --------------------------------------------------------------------------------------------------

import warnings; warnings.simplefilter('ignore')

def get_index_from_input_movie(user_input):
    return dataset[dataset['title'].str.lower().str.replace('-', '').str.replace('the', '').str.replace(':', '').str.strip() == user_input]['index'].values[0]
    
def stop_and_stem(uncleaned_list):
    ps = PorterStemmer()
    stop = set(stopwords.words('english'))
    stopped_list = [i for i in uncleaned_list if i not in stop]
    stemmed_words = [ps.stem(word) for word in stopped_list]
    return stemmed_words

def remove_stopwords(uncleaned_list):
    stop = set(stopwords.words('english'))
    stopped_list = [i for i in uncleaned_list if i not in stop]
    return stopped_list

def remove_punctuation(a_list):
    for i, text in enumerate(a_list):
        for ch in ['\\','`','*','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'','?','/',"'s"]:
            if ch in text:
                a_list[i] = a_list[i].replace(ch,'')
    return a_list

def search_words(row, list_of_words):
    ps = PorterStemmer()
    row = [ps.stem(x) for x in row]
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

def find_correct_movie(user_input, movie_list):
    scores_similarity=[]

    for item in movie_list:
        ed = nltk.edit_distance(user_input, item)
        scores_similarity.append(ed)
    correct_movie_index = scores_similarity.index(min(scores_similarity))
    correct_movie = movie_list[correct_movie_index].lower()
    return correct_movie

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list

# I will use either the intersection or the union function. I will decide later which of the two.

def replace_letter_with_number(ll):
    for j in range(len((ll))):
        if ll[j].startswith(('0', '1', '2', '3', '4','5','6', '7', '8', '9')):
            a = re.sub('[^0-9]', '',ll[j]) 
            ll[j] = a
        else:
            pass
    return ll


# -----------------------------------------------------------------------------------------------


# Import the dataset

# dataset = pd.read_pickle('C:\\Users\\dq186sy\\Desktop\\Big Data Content Analytics\\Movie Recommendation System\\dataset_embedded_02092019.pkl')

dataset = pd.read_pickle(os.path.join(os.getcwd(), 'pickled_data_per_part\\dataset_part_4_07032020.pkl'))

dataset = dataset.reset_index()

dataset['index'] = np.arange(0, len(dataset))

# It is important to reset the index of the dataset in order to get the correct index per movie!

# -------------------------------------------------------------------------------------------------

def recommend_movie(input_one, input_two, input_movie):

    # Create the movieovie_genre list with the unique types of genre 

    with open(os.path.join(os.getcwd(), 'pickled_data_per_part\\genres_list_16022020.pkl'), 'rb') as f:
        movie_genre_list = pickle.load(f)

    movie_genre_list = [x.lower() for x in movie_genre_list]


    # -------------------------------------------------------------------------------------------------


    # Phase 1: Get the user's input and transform it to the appropriate form

    # First Input
    input_one = find_correct_genre(input_one.lower(), movie_genre_list)

    # Second Input
    input_movie = input_movie.lower().replace('-', '').replace('the', '').replace(':', '').strip()

    # Third Input
    input_two = input_two.lower().replace(',', '').replace('.', '').split(' ')

    inputs_list = remove_stopwords(input_two)


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
        
        input_movie = find_correct_movie(input_movie, selected_genre_movies_list) #probably there is no reason in ckecking this!
        
        assert input_movie in selected_genre_movies_list
        
        # Isolate the movie plot of the movie provided from the user [If the movie is part of the dataset].

        movie_plot_new = locked_frame.loc[:, 'plot'].loc[(locked_frame['title'].str.lower().str.replace('-', '').str.replace('the', '').str.replace(':', '').str.strip() == input_movie)].apply(lambda x: list(set(re.split(' |,|\n', x.strip().lower())))).values[0]
        
        cleaned_movie_plot = remove_stopwords(movie_plot_new)

        plot_user_input_list = inputs_list + cleaned_movie_plot
        
        plot_user_input_list = remove_punctuation(plot_user_input_list)

        plot_user_input_list = [x for x in plot_user_input_list if x]

        plot_user_input_list = list(dict.fromkeys(plot_user_input_list))
        
        plot_user_input_list = replace_letter_with_number(plot_user_input_list)


        # -------------------------------------------------------------------------------------------------


        # Get the index of the movie provied by the user

        movie_index = get_index_from_input_movie(input_movie)
        
        locked_frame_index = locked_frame.loc[locked_frame['title'].str.lower().str.replace('-', '').str.replace('the', '').str.replace(':', '').str.strip() == input_movie]['index'].values[0]
        
        # -------------------------------------------------------------------------------------------------

        # Phase 3: Keeping the words with meaning
        
        # What I have noticed is that many words included in both Plot Summary and Combined Features were not capable to discremenate 
        # a movie. Thus, I tried to keep only the words that could better differenciate a movie from its peer.
        
        # Tfidf vectorizer
        
        tfv = TfidfVectorizer(use_idf=True)
        
        x = tfv.fit_transform(locked_frame['movie_features'])

        
        # get the vector of the matched movie
        
        vector_tfidfvectorizer=x[locked_frame_index]
     
        
        # place tf-idf values in a pandas data frame
        
        df = pd.DataFrame(vector_tfidfvectorizer.T.todense(), index=tfv.get_feature_names(), columns=["Importance/word"])
        
        df = df.sort_values(by=["Importance/word"], ascending=False)
        
        df = df[df.loc[:,'Importance/word']>0.1]
        
        list_of_words = df.loc[:,'Importance/word'].index.tolist()

        
        plot_user_input_list_new = union(list_of_words, plot_user_input_list)
        
        # Having the list of words to keep, it is useful to store those words in order to use the later!
        
        # -------------------------------------------------------------------------------------------------
        
        # Phase 4: Locate the word embeddings belonging to each of the three different columns (Actors, Plot, Features)
        
        # Phase 4.1: Locate the embeddings of the movie selected by the user!
        
        # Get Casting Embeddings based on the movie_index

        cast_vector_average = dataset['average_cast_vectors'][dataset['index'] == movie_index]
        cast_vector_min = dataset['minimum_cast_vectors'][dataset['index'] == movie_index]
        cast_vector_max = dataset['maximum_cast_vectors'][dataset['index'] == movie_index]
        
        cast_vector = np.hstack([cast_vector_average.apply(pd.Series).values,
                                 cast_vector_min.apply(pd.Series).values,
                                 cast_vector_max.apply(pd.Series).values])
        
        # Get Plot Embeddings based on the movie_index

        plot_vector_average = dataset['average_plot_vectors'][dataset['index'] == movie_index]
        plot_vector_min = dataset['minimum_plot_vectors'][dataset['index'] == movie_index]
        plot_vector_max = dataset['maximum_plot_vectors'][dataset['index'] == movie_index]
        
        plot_vector = np.hstack([plot_vector_average.apply(pd.Series).values,
                                 plot_vector_min.apply(pd.Series).values,
                                 plot_vector_max.apply(pd.Series).values])

        # Get Features Embeddings based on the movie_index

        feature_vector_average = dataset['average_combined_features_vectors'][dataset['index'] == movie_index]
        feature_vector_min = dataset['minimum_combined_features_vectors'][dataset['index'] == movie_index]
        feature_vector_max = dataset['maximum_combined_features_vectors'][dataset['index'] == movie_index]
        
        feature_vector = np.hstack([feature_vector_average.apply(pd.Series).values,
                                    feature_vector_min.apply(pd.Series).values,
                                    feature_vector_max.apply(pd.Series).values])

        # Get Reviews Embeddings based on the movie_index

        reviews_vector_average = dataset['average_reviews_vectors'][dataset['index'] == movie_index]
        reviews_vector_min = dataset['minimum_reviews_vectors'][dataset['index'] == movie_index]
        reviews_vector_max = dataset['maximum_reviews_vectors'][dataset['index'] == movie_index]
        
        reviews_vector = np.hstack([reviews_vector_average.apply(pd.Series).values,
                                    reviews_vector_min.apply(pd.Series).values,
                                    reviews_vector_max.apply(pd.Series).values])

        
        # Phase 4.2: Locate the embeddings of the movies that match the GENRE given by the user (i.e the embeddings of all the ACTION movies)
        
        # Load the saved embeddings trained by the multi-input keras classifier
        
        with open(os.path.join(os.getcwd(), 'model_one\\keras_embeddings_array_concatenated_07032020.pkl'), 'rb') as f:
            keras_embeddings_array_concatenated = pickle.load(f)
        
        # Indexes list is a list of the index each "genre" movie has in the locked dataframe (i.e the index of all the action movies)
        
        locked_movie_embeddings = keras_embeddings_array_concatenated[indexes_list]
        

        # -------------------------------------------------------------------------------------


        # Concatenate the embeddings

        selected_movie_embeddings = np.hstack([cast_vector, plot_vector, feature_vector, reviews_vector])
        
        assert selected_movie_embeddings.shape[1] == locked_movie_embeddings.shape[1]
        
        # The dimension of those two arrays should be the same.
        
        # Calculate Cosine Distance

        cosine_dist = cosine_distances(locked_movie_embeddings, selected_movie_embeddings.reshape(1,-1))

        
        # Get the similar movies & Slice the dataframe on the top 5 most similar movies to the movie given  by the user

        movie_return = np.argsort(cosine_dist, axis=None).tolist()[1:6]

        # movie_return contains the index of the 5 movies most similar to the movie selected by the user!
        
        # So the next step is to isolate those 5 movies and their features
        
        locked_frame_new = locked_frame[locked_frame.loc[:, 'index'].isin(movie_return)]


        # -------------------------------------------------------------------------------------

        # Phase 5: Create two new columns "Unique Words" + "Number of words"

        # Create the new column of "UNIQUE" words of the combined features
        
        locked_frame_new.loc[:, 'unique_words'] = locked_frame_new.loc[:, 'movie_features'].apply(lambda x: list(set(re.split(' |,|\n', x.strip().lower()))))

        locked_frame_new.loc[:, 'unique_words'] = locked_frame_new.loc[:, 'unique_words'].apply(lambda x: remove_punctuation(x))

        locked_frame_new.loc[:, 'unique_words'] = [[x for x in lst if x] for lst in locked_frame_new.loc[:, 'unique_words']]
      
        # Create the column "Number of words" for each word contained in the unique words column

        locked_frame_new.loc[:, 'number_of_words'] = locked_frame_new.loc[:, 'unique_words'].apply(search_words, args=(plot_user_input_list_new,))


        # -------------------------------------------------------------------------------------

        
        # Phase 6.1: Recommend to the user the three most similar and highly scored movies 
        
        
        # Calculate the movie score

        primary_genre = list((locked_frame_new.loc[:, "lower_case_genres"].map(lambda x: input_one in x)*0.2))

        locked_frame_new.loc[:, 'movie_score'] = 0.1*locked_frame_new.loc[:, 'imdb_rating'].astype(float) + 0.5*locked_frame_new.loc[:, 'number_of_words'] + 0.2*locked_frame_new.loc[:, "sentiment_value"]

        locked_frame_new.loc[:, 'movie_score'] = locked_frame_new.loc[:, 'movie_score'] + primary_genre[0]


        # ---------------------------------------------------------------------------------------


        # Give to the user the proper movie recommendation

        top_three_rows = locked_frame_new.nlargest(3, 'movie_score')
        
        # top_three_rows.rename(columns={'movie_title':'Movie Title', 'updated_rating':'IMDB Rate', 'movie_imdb_link':"Movie's Link"}, inplace=True)

        # Recommend the movie

        recommendations_list = top_three_rows.loc[:, ['title', 'imdb_rating', 'imdb_url']].values.tolist()
        
        return recommendations_list
        
    else:
        
        plot_user_input_list = inputs_list
        
        locked_frame.loc[:, 'unique_words'] = locked_frame.loc[:, 'movie_features'].apply(lambda x: list(set(re.split(' |,|\n', x.strip().lower()))))

        locked_frame.loc[:, 'unique_words'] = locked_frame.loc[:, 'unique_words'].apply(lambda x: remove_punctuation(x))

        locked_frame.loc[:, 'unique_words'] = [[x for x in lst if x] for lst in locked_frame.loc[:, 'unique_words']]
      
        # Create the column "Number of words" for each word contained in the unique words column

        locked_frame.loc[:, 'number_of_words'] = locked_frame.unique_words.apply(search_words, args=(plot_user_input_list,))

        
        
        # Phase 6.2: Recommend to the user the three most similar and highly scored movies
        
        
        primary_genre = list((locked_frame.loc[:, "lower_case_genres"].map(lambda x: input_one in x)*0.2))

        locked_frame.loc[:, 'movie_score'] = 0.1*locked_frame.loc[:, 'imdb_rating'].astype(float) + 0.5*locked_frame.loc[:, 'number_of_words'] + 0.2*locked_frame.loc[:, 'sentiment_value']

        locked_frame.loc[:, 'movie_score'] = locked_frame.loc[:, 'movie_score'] + primary_genre[0]
        
        
        # Give to the user the proper movie recommendation

        top_three_rows = locked_frame.nlargest(3, 'movie_score')
        
        # top_three_rows.rename(columns={'movie_title':'Movie Title', 'updated_rating':'IMDB Rate', 'movie_imdb_link':"Movie's Link"}, inplace=True)

        
        # Recommend the movie

        recommendations_list = top_three_rows.loc[:, ['title', 'imdb_rating', 'imdb_url']].values.tolist()
        
        return recommendations_list