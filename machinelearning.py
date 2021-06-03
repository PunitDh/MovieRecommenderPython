import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

data = pd.read_csv("tmdb_5000_movies.csv").fillna('')
data['keywords'] = data['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x]).apply(lambda x: " ".join(map(str, x)))
moviedata = data['title'] + " " + data['overview'] + " " + data['tagline'] + " " + data['keywords']

vectorizer = TfidfVectorizer().fit_transform(moviedata)
similarity_table = linear_kernel(vectorizer)

def get_recommendations(title):
    idx = pd.Series(data.index, index=data['title'])[title]
    similar_movies = sorted(list(enumerate(similarity_table[idx])), key=lambda x: x[1], reverse=True)
    return data['title'][[i[0] for i in similar_movies[1:11]]]

while True:
    movie = input("Please enter the name of the last movie you watched: ")
    try:
        print(get_recommendations(movie))
    except KeyError:
        print("Invalid movie.")