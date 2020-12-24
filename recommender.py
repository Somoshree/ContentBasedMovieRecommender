import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
creditsdf=pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')
credits_colrenamed=creditsdf.rename(index=str,columns={"movie_id":"id"})
movies_merged=movies.merge(credits_colrenamed,on="id")
movies_cleaned = movies_merged.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
movies_cleaned['original_title'] = movies_cleaned['original_title'].str.lower()

tfv= TfidfVectorizer(min_df=3,max_features=None,strip_accents='unicode',analyzer='word',token_pattern='\w{1,}',
                    ngram_range=(1,3),stop_words='english')
movies_cleaned['overview']=movies_cleaned['overview'].fillna('')
#pickle.dump(movies_cleaned,open('movies.pkl','wb'))
tfv_matrix=tfv.fit_transform(movies_cleaned['overview'])
#pickle.dump(tfv,open('transform.pkl','wb'))

sig=sigmoid_kernel(tfv_matrix,tfv_matrix)
#pickle.dump(sig,open('sigmoidkernel.pkl','wb'))
# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title'])
#pickle.dump(indices,open('indices.pkl','wb'))
def give_rec(title, sig=sig):
    title=title.lower()
    if title not in movies_cleaned['original_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        
        # Get the index corresponding to original_title
        idx = indices[title]
        if(type(idx)!=np.int64):
            idx=indices[title][0]
        # Get the pairwsie similarity scores 
        sig_scores = list(enumerate(sig[idx]))
    
        # Sort the movies 
        sig_scores = sorted(sig_scores,key=lambda x: x[1], reverse=True)
    
        # Scores of the 10 most similar movies
        sig_scores = sig_scores[1:11]
    
        # Movie indices
        movie_indices = [i[0] for i in sig_scores]
        # Top 10 most similar movies
        l=[]
        for m in movie_indices:
            l.append(movies_cleaned['original_title'].iloc[m])
        return l

app= Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = give_rec(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')
    
if __name__ == '__main__':
    app.run()