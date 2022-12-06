import pandas as pd
from jikanpy import Jikan
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import yaml
import ast
import difflib
import requests
from bs4 import BeautifulSoup
import re
import wikipedia

jikan = Jikan()
columns = ['type','mal_id','title_english','popularity','rating','genres','studios','themes','demographics','synopsis']
file = './top_anime.csv'
pd.options.mode.chained_assignment = None  # default='warn'

def cell_splitter(temp_df):
    for i, values in enumerate(temp_df):
        if isinstance(values, str):
            values = yaml.load(values,Loader=yaml.Loader)
        value_list = []
        for value in values:
            value_list.append(value['name'])
        if len(values) == 1:
            value_list = value_list[0]
        temp_df.iloc[i] = value_list

    return temp_df

def nlp(text, num_of_words=10):
    text = text.values[0]
    cv = CountVectorizer(stop_words='english')
    dtm = cv.fit_transform([text])
    # Build LDA Model with GridSearch params
    lda_model = LatentDirichletAllocation(n_components=1,
                                        learning_decay=0.5,
                                        max_iter=50,
                                        learning_method='online',
                                        random_state=50,
                                        batch_size=100,
                                        evaluate_every = -1,
                                        n_jobs = -1)
    lda_output = lda_model.fit_transform(dtm)
    for topic in lda_model.components_:
        words = [cv.get_feature_names_out()[i] for i in topic.argsort()][-num_of_words:]
    return words

def wiki(title):
    try:
        title = title + ' TV Series'
        page = str(wikipedia.page(title).content)
    except:
        page = str(wikipedia.page(title).content)
    plotStr = page.split('Plot ==')[1].split('\n\n\n== ')[0]
    plotStr = re.sub('\\n.*?',"",plotStr)
    plotStr = re.sub("\\.*?","",plotStr)
    return plotStr

def nlp(text, num_of_words=10):
    text = text.values[0]
    cv = CountVectorizer(stop_words='english')
    dtm = cv.fit_transform([text])
    # Build LDA Model with GridSearch params
    lda_model = LatentDirichletAllocation(n_components=1,
                                        learning_decay=0.5,
                                        max_iter=50,
                                        learning_method='online',
                                        random_state=50,
                                        batch_size=100,
                                        evaluate_every = -1,
                                        n_jobs = -1)
    lda_output = lda_model.fit_transform(dtm)
    for topic in lda_model.components_:
        words = [cv.get_feature_names_out()[i] for i in topic.argsort()][-num_of_words:]
    return words

def anime_search(search,df):
    if df.empty:
        results = jikan.search('anime', search)
        df = pd.DataFrame(results['data'])
    else:
        search = search.lower()
        df['title_english'] = df['title_english'].str.lower()
        df = df[df['title_english'].str.contains(search, na=False)]
    temp_df = df[columns].dropna()
    temp_df = temp_df[temp_df['type'] == 'TV']
    temp_df.drop('type',axis=1,inplace=True)
    temp_df.reset_index(drop=True)
    temp_df['genres'] = cell_splitter(temp_df['genres'])
    temp_df['studios'] = cell_splitter(temp_df['studios'])
    temp_df['themes'] = cell_splitter(temp_df['themes'])
    temp_df['demographics'] = cell_splitter(temp_df['demographics'])

    temp_df['synopsis'].iloc[0] = nlp(temp_df['synopsis'])

    return temp_df

def top_anime(pages=80):
    if pages==80 and os.path.exists(file):
        df = pd.read_csv(file,index_col=0)
    else:
        df = pd.DataFrame()
        for i in range(pages):
            results = jikan.top(type='anime',page=i)
            new_df = pd.DataFrame(results['data'])
            df = pd.concat([df,new_df], ignore_index=True)
            time.sleep(1)
        df.to_csv(file)

    pop_df = df[columns].dropna()
    pop_df = pop_df[pop_df['type'] == 'TV']
    pop_df.drop('type',axis=1,inplace=True)
    pop_df['genres'] = cell_splitter(pop_df['genres'])
    pop_df['studios'] = cell_splitter(pop_df['studios'])
    pop_df['themes'] = cell_splitter(pop_df['themes'])
    pop_df['demographics'] = cell_splitter(pop_df['demographics'])
    for j, synopses in enumerate(pop_df['synopsis']):
        try:
            synopsis = wiki(pop_df['title_english'].iloc[j])
        except:
            synopsis = synopses
        try:
            pop_df['synopsis'].iloc[j] = nlp(pd.Series(synopsis))
        except:
            pop_df['synopsis'].iloc[j] = []
        time.sleep(1)

    return pop_df

def get_cluster(fav_anime, n_clusters=38):
    # fav_df = pd.DataFrame(columns = columns)
    # fav_df.drop('type',axis=1,inplace=True)
    # fav_anime = ['fullmetal alchemist: brotherhood','dragon ball z','steins;gate','psycho-pass','my hero academia','code geass: lelouch of rebellion','rurouni kenshin','attack on titan','puella magi madoka magica','one punch man']
    # search_df = None
    # sleep_timer = 1
    # if os.path.exists(file):
    #     search_df = pd.read_csv(file,index_col=0)
    #     sleep_timer = 0
    # for fav in fav_anime:
    #     df = anime_search(fav, search_df)
    #     df.sort_values('popularity')
    #     temp_df = df.iloc[[0]]
    #     fav_df = pd.concat([fav_df,temp_df], ignore_index=True)
    #     time.sleep(sleep_timer)

    # Get df of popular anime & corresponding info
    pop_df_file = './pop_df.csv'
    if os.path.exists(pop_df_file):
        pop_df = pd.read_csv(pop_df_file,index_col=0)
    else:
        pop_df = top_anime()
        pop_df = pop_df.drop_duplicates(subset=['title_english']).reset_index(drop=True)
        pop_df.to_csv(pop_df_file)

    # indices = []
    # for id in fav_df['mal_id']:
    #     index = pop_df[pop_df['mal_id']==id].index[0]
    #     indices.append(index)

    # Convert pop_df into a dummy dataframe with only 0s or 1s values
    mlb = MultiLabelBinarizer()
    ml_df = pd.get_dummies(pop_df,columns=['rating'])
<<<<<<< HEAD
    genres = pd.DataFrame(mlb.fit_transform(pop_df['genres'].apply(lambda x: ast.literal_eval(x))),columns=mlb.classes_)
    studios = pd.DataFrame(mlb.fit_transform(pop_df['studios'].apply(lambda x: ast.literal_eval(x))),columns=mlb.classes_)
    themes = pd.DataFrame(mlb.fit_transform(pop_df['themes'].apply(lambda x: ast.literal_eval(x))),columns=mlb.classes_)
    demographics = pd.DataFrame(mlb.fit_transform(pop_df['demographics'].apply(lambda x: ast.literal_eval(x))),columns=mlb.classes_)
    #synopsis = pd.DataFrame(mlb.fit_transform(pop_df['synopsis'].apply(lambda x: ast.literal_eval(x))),columns=mlb.classes_)
    dummy_df = pd.concat([ml_df,genres,studios,themes,demographics], axis=1).drop(['score','popularity','mal_id','title_english','genres','studios','themes','demographics','synopsis'], axis=1).fillna(0)
=======
    genres = pd.DataFrame(mlb.fit_transform(pop_df['genres'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
    studios = pd.DataFrame(mlb.fit_transform(pop_df['studios'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
    themes = pd.DataFrame(mlb.fit_transform(pop_df['themes'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
    demographics = pd.DataFrame(mlb.fit_transform(pop_df['demographics'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
    synopsis = pd.DataFrame(mlb.fit_transform(pop_df['synopsis'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
    dummy_df = pd.concat([ml_df,genres,studios,themes,demographics,synopsis], axis=1).drop(['popularity','mal_id','title_english','genres','studios','themes','demographics','synopsis'], axis=1).fillna(0)
>>>>>>> streamlit

    # Normalizing dummy_df to be between 0 & 1
    normalizer = MinMaxScaler()
    normal = normalizer.fit_transform(dummy_df)
    normal_df = pd.DataFrame(data=normal,columns=dummy_df.columns)

    # Clustering shows together based on number of clusters and categories
    model = AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True)
    y = model.fit_predict(normal_df)

    # Combining clusters with pop_df to select only related shows
    pop_df['cluster'] = y
    # display(pop_df.iloc[indices])
    cluster = pop_df[pop_df['title_english']==fav_anime]['cluster'].values[0]
    cluster_df = pop_df[pop_df['cluster']==cluster]
    return cluster_df
