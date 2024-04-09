import pandas as pd

unames = ['user_id','gender','age','occupation','zip']
rnames = ['user_id','movie_id','rating','timestamp']
mnames = ['movie_id','title','genres']

user = pd.read_csv('users.dat', sep='::', header=None, names=unames, engine='python', encoding="ISO-8859-1")
ratings = pd.read_csv('ratings.dat', sep='::', header=None, names=rnames, engine='python', encoding="ISO-8859-1")
movies = pd.read_csv('movies.dat', sep='::', header=None, names=mnames, engine='python', encoding="ISO-8859-1")
# Merge the datasets
data = pd.merge(pd.merge(ratings, movies), user)


data.to_csv('ml-1m.csv', index=False)