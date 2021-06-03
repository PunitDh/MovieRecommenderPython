import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise

data1 = pd.read_csv("tmdb_5000_credits.csv")
data2 = pd.read_csv("tmdb_5000_movies.csv")
data1.columns = ['id', 'title', 'cast', 'crew']
# data2 = data2.merge(data1, on='id')
data2['overview'] = data2['overview'].fillna('')

C = data2['vote_average'].mean()
m = data2['vote_count'].quantile(0.9)
q_movies = data2.copy().loc[data2['vote_count'] >= m]


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R) + (m/(m+v)*C)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)


vectorizer = TfidfVectorizer()
vectorizer_matrix = vectorizer.fit_transform(data2['overview'])
similarity_matrix = linear_kernel(vectorizer_matrix, vectorizer_matrix)
indices = pd.Series(data2.index, index=data2['original_title']).drop_duplicates()


def get_recommendations(title, similarity_matrix = similarity_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data2['original_title'].iloc[movie_indices]




while True:
    movie = input("Please enter the name of a movie: ")
    try:
        print(get_recommendations(movie))
    except KeyError:
        print("Invalid movie.")