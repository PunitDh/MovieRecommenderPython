#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

df1 = pd.read_csv("tmdb_5000_credits.csv")
df2 = pd.read_csv("tmdb_5000_movies.csv")
df1.columns = ['id','title','cast','crew']
df2 = df2.merge(df1,on='id')
C = df2['vote_average'].mean()
m = df2['vote_count'].quantile(0.9)
q_movies = df2.copy().loc[df2['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R) + (m/(m+v)*C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['original_title']).drop_duplicates()

def get_recommendations(title, cosine_sim = cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['original_title'].iloc[movie_indices]

# get_recommendations("The Dark Knight Rises")


# In[10]:


from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


# In[11]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[12]:


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    
    return []


# In[13]:


df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

df2[['original_title','cast','director','keywords','genres']].head(3)


# In[15]:


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
    df2[feature] = df2[feature].apply(clean_data)


# In[16]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['original_title'])

# print(get_recommendations("The Dark Knight Rises", cosine_sim2))


# In[22]:


# from surprise import Reader, Dataset, SVD
# from surprise.model_selection import cross_validate
# from surprise.model_selection import KFold
#
#
# reader = Reader()
# ratings = pd.read_csv('ratings_small.csv')
# # print(ratings.head())
#
#
# # In[ ]:
# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# kf = KFold(n_splits=5)
# kf.split(data)
# # data.split(n_folds=5)
#
# svd = SVD()
# cross_validate(svd, data, measures=['RMSE','MAE'])
#
# trainset = data.build_full_trainset()
# svd.fit(trainset)
#
# print(ratings[ratings['userId'] == 1])
#
# print(svd.predict(1, 302, 3))

while True:
    movie = input("Please enter the last movie you watched: ")
    try:
        print(get_recommendations(movie,cosine_sim2))
    except KeyError:
        print("Invalid movie name.")