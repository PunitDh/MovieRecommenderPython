import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

data = pd.read_csv("tmdb_5000_movies.csv").fillna('')
data['keywords'] = data['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x]).apply(lambda x: " ".join(map(str,x)))
moviedata = data['title'] + " " + data['tagline'] + " " + data['overview'] + " " + data['keywords']

vectorizer = TfidfVectorizer().fit_transform(moviedata)
similarity_table = linear_kernel(vectorizer)

def get_recommendations(title):
    list_of_movies = pd.Series(data.index, index=data['title'])
    similarity_scores = sorted(list(enumerate(similarity_table[list_of_movies[title]])), key=lambda x: x[1], reverse=True)
    return data['title'][[i[0] for i in similarity_scores[1:11]]]

while True:
    movie = input("Please enter the last movie you watched: ")
    try:
        print( get_recommendations(movie) )
    except KeyError:
        print("Invalid movie.")