#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np 
import pandas as pd


# In[122]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv") 


# In[153]:


credits.loc[:,'cast':'crew']


# In[154]:


movies.loc[:,'genres']


# In[123]:


movies.head()


# In[124]:


credits.head()


# In[125]:


print("movies shape : ",movies.shape)
print("credits shape : ",credits.shape)


# In[126]:


movies.info()


# In[127]:


credits.info()


# In[128]:


movies = movies.merge(credits,on='title')
movies


# In[129]:


movies.shape


# In[130]:


movies =movies[['title','overview','genres','keywords','cast','crew']]
movies.head()


# In[131]:


movies.info()


# In[132]:


import ast


# In[133]:


def convert(text):
    L=[]
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[134]:


movies.isnull().sum()


# In[135]:


movies.dropna(inplace=True)


# In[136]:


import ast
movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[137]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[138]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i[''])
        counter+=1
    return L 


# In[139]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[59]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[60]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[61]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[62]:


movies.sample(5)


# In[63]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[64]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[65]:


movies.head()


# In[66]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()


# In[67]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']


# In[68]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[69]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[71]:


vector = cv.fit_transform(new['tags']).toarray()
vector


# In[72]:


vector.shape


# In[73]:


from sklearn.metrics.pairwise import cosine_similarity


# In[74]:


similarity=cosine_similarity(vector)


# In[75]:


similarity


# In[76]:


similarity.shape


# In[115]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)  


# In[116]:


recommend('Gandhi')


# In[117]:


recommend("Pirates of the Caribbean: At World's End")


# In[118]:


recommend('Jurassic Park')


# In[ ]:




