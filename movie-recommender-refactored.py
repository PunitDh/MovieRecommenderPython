import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer

data1 = pd.read_csv("tmdb_5000_credits.csv")
data = pd.read_csv("tmdb_5000_movies.csv")
data1.columns = ['id', 'title', 'cast', 'crew']
data = data.merge(data1, on='id')
C = data['vote_average'].mean()
m = data['vote_count'].quantile(0.9)
movies = data.copy().loc[data['vote_count'] >= m]


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


movies['score'] = movies.apply(weighted_rating, axis=1)
movies = movies.sort_values('score', ascending=False)
features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:
    data[feature] = data[feature].apply(literal_eval)


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


data['director'] = data['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    data[feature] = data[feature].apply(get_list)



def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    data[feature] = data[feature].apply(clean_data)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


data['soup'] = data.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(data.index, index=data['original_title'])


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['original_title'].iloc[movie_indices]

while True:
    movie = input("Please enter the last movie you watched: ")
    try:
        print(get_recommendations(movie))
    except KeyError:
        print("Invalid movie name.")