# Run the algorithm through webhook

# import flask dependencies --------------------------------------------------------------------------------------------------

from flask import Flask, request, make_response, jsonify

from movie_recommendation_v13 import recommend_movie

import sys

# initialize the flask app
app = Flask(__name__)

# default route
@app.route('/')

def index():
    return 'Chatbot is online and running!'

res = None

# Function 1 --------------------------------------------------------------------------------------------------

def suggest_movie(req):

    global res

    # fetch action from json
    
    user_genre = req.get('queryResult').get('parameters').get('user_genre')

    user_prefrences = req.get('queryResult').get('parameters').get('user_prefrences')
    
    user_movie = req.get('queryResult').get('parameters').get('user_movie')

    res = recommend_movie(user_genre, user_prefrences, user_movie)

    return {'fulfillmentText': "Thank you! Based on your preferences, I propose the following two movies:\n\n{0} \n(IMDB Rating: {1} & Link: {2}) \n\n{3} \n(IMDB Rating: {4} & Link: {5})\n\nHave you seen any of these movies?".format(res[0][0], res[0][1], res[0][2], res[1][0], res[1][1], res[1][2])}

# Function 2 --------------------------------------------------------------------------------------------------

def propose_second_movie(req):

    ask_seen = req.get('queryResult').get('parameters').get('ask_seen_one')

    if ask_seen == 'yes' or ask_seen == 'Yes':
        
        return {'fulfillmentText': 'Then I propose you to see this movie: \n\n{0} \n(IMDB Rating: {1} & Link: {2}) \n\n{3} \n(IMDB Rating: {4} & Link: {5})\n\n(Type "Thank you" if would like to end the conversation!)'.format(res[2][0], res[2][1], res[2][2], res[3][0], res[3][1], res[3][2])}

    elif ask_seen == 'no' or ask_seen == 'No':
        
        return {'fulfillmentText': 'Great news! Grab a bowl of Pop-Corn and enjoy one of the two films!\n\n(Type "Thank you" if would like to end the conversation!)'}

# Function 3 --------------------------------------------------------------------------------------------------

def thanks(req):

    greeding = req.get('queryResult').get('parameters').get('greeding')

    if greeding == 'Thank you' or greeding == 'thank you' or greeding == 'Thanks' or greeding == 'thanks' or greeding == 'Nice' or greeding == 'nice':

        return {'fulfillmentText': "You're welcome! :)\n\nWould you like to choose another movie? (Yes/No)"}

# Function 4 --------------------------------------------------------------------------------------------------

def rerun_conversation(req):

    rerun_conversation_text=req.get('queryResult').get('parameters').get('rerun_again')

    if rerun_conversation_text == "No" or rerun_conversation_text == "no":

        return {'fulfillmentText': "Enjoy your film!"}

    elif rerun_conversation_text == "Yes" or rerun_conversation_text == "yes":

        return {"followupEventInput": {"name": "KellyMovieBot","languageCode": "en-US"}}

# Create a route for webhook  --------------------------------------------------------------------------------------------------

# For webhook to properly work, I have to copy paste the https link from ngrok

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():

    # return response
    req = request.get_json(force=True, silent = True)

    action = req.get('queryResult').get('action')

    intent_name = req.get('queryResult').get('intent').get('displayName')

    if action == "get_results" and intent_name == 'KellyMovieBot':

        return make_response(jsonify(suggest_movie(req)))

    elif action == "ask_question" and intent_name == 'ask_if_seen_movies':

        return make_response(jsonify(propose_second_movie(req)))

    elif action == "thanks_giving_end" and intent_name == "thanks_giving":

        return make_response(jsonify(thanks(req)))

    elif action == "rerun_conversation_text" and intent_name == "rerun_conversation":

        return make_response(jsonify(rerun_conversation(req)))

# run the app
if __name__ == '__main__':
   app.run(port=9090, debug=True)