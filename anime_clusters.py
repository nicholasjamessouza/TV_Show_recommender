import pandas as pd
from jikanpy import Jikan
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
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

def get_cluster(fav_anime, sort='Closest Match'):

    final_df_file = './final_df.csv'
    if os.path.exists(final_df_file):
        final_df = pd.read_csv(final_df_file,index_col=0)
    else:
        # Get df of popular anime & corresponding info
        pop_df_file = './pop_df.csv'
        if os.path.exists(pop_df_file):
            pop_df = pd.read_csv(pop_df_file,index_col=0)
        else:
            pop_df = top_anime()
            pop_df = pop_df.drop_duplicates(subset=['title_english']).reset_index(drop=True)
            pop_df.to_csv(pop_df_file)

        # Convert pop_df into a dummy dataframe with only 0s or 1s values
        mlb = MultiLabelBinarizer()
        ml_df = pd.get_dummies(pop_df,columns=['rating'])
        genres = pd.DataFrame(mlb.fit_transform(pop_df['genres'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
        studios = pd.DataFrame(mlb.fit_transform(pop_df['studios'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
        themes = pd.DataFrame(mlb.fit_transform(pop_df['themes'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
        demographics = pd.DataFrame(mlb.fit_transform(pop_df['demographics'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
        synopsis = pd.DataFrame(mlb.fit_transform(pop_df['synopsis'].apply(lambda x: ast.literal_eval(x) if ',' in x else [x])),columns=mlb.classes_)
        dummy_df = pd.concat([ml_df,genres,studios,themes,demographics,synopsis], axis=1).drop(['popularity','mal_id','title_english','genres','studios','themes','demographics','synopsis'], axis=1).fillna(0)

        # Normalizing dummy_df to be between 0 & 1
        normalizer = MinMaxScaler()
        normal = normalizer.fit_transform(dummy_df)
        normal_df = pd.DataFrame(data=normal,columns=dummy_df.columns)

        # Clustering shows together based on number of clusters and categories
        model = AgglomerativeClustering(n_clusters=38, compute_distances=True)
        y = model.fit_predict(normal_df)

        # Converting high dimension dataframe into 3dimensions for visualization and proximity
        plotX = normal_df
        plotX['Cluster'] = y
        plotX.columns = normal_df.columns
        pca_3d = PCA(n_components=3)
        PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))
        PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
        plotX = pd.concat([plotX,PCs_3d], axis=1, join='inner')

        # Finding nearest neighbors based on distance in 3D space
        neighbors = []
        for i in range(len(PCs_3d)):
            PCs = PCs_3d.drop(i)
            neighbor = []
            p1 = np.array([PCs_3d['PC1_3d'].iloc[i], PCs_3d['PC2_3d'].iloc[i], PCs_3d['PC3_3d'].iloc[i]])
            for j in range(len(PCs)):

                p2 = np.array([PCs['PC1_3d'].iloc[j], PCs['PC2_3d'].iloc[j], PCs['PC3_3d'].iloc[j]])
                squared_dist = np.sum((p1-p2)**2, axis=0)
                dist = np.sqrt(squared_dist)
                neighbor.append(dist)

            current_cluster = plotX['Cluster'].iloc[i]
            current_cluster_list = plotX[plotX['Cluster']==current_cluster].drop(i).index.to_list()
            neighbor_df = pd.Series(neighbor)
            neighbor_df.index = PCs.index
            neighbor_df = neighbor_df[current_cluster_list]
            neighbor_df = neighbor_df.sort_values()
            neighbor_list = neighbor_df.index.to_list()
            neighbors.append(neighbor_list)
        pop_df['neighbor'] = neighbors

        # Combining clusters with pop_df to select only related shows
        pop_df['cluster'] = y
        final_df = pop_df

    top_anime = pd.read_csv(file,index_col=0)
    top_anime['images_url'] = top_anime['images'].apply(lambda x: x.split("url': '")[1].split("'")[0])
    img_url = top_anime[['mal_id','images_url']]
    final_df = final_df.merge(img_url, on='mal_id').drop_duplicates(subset=['mal_id']).reset_index(drop=True)
    img_url = final_df[final_df['title_english']==fav_anime]['images_url'].values[0]
    if sort == 'Score':
        idx = final_df[final_df['title_english']==fav_anime].index
        cluster = final_df[final_df['title_english']==fav_anime]['cluster'].values[0]
        cluster_df = final_df[final_df['cluster']==cluster]
        cluster_df.drop(idx, inplace=True)
    elif sort == 'Popularity':
        idx = final_df[final_df['title_english']==fav_anime].index
        cluster = final_df[final_df['title_english']==fav_anime]['cluster'].values[0]
        cluster_df = final_df[final_df['cluster']==cluster].sort_values(by='popularity')
        cluster_df.drop(idx, inplace=True)
    elif sort == 'Closest Match':
        cluster = final_df[final_df['title_english']==fav_anime]['neighbor'].values[0]
        cluster = ast.literal_eval(cluster)
        cluster_df = final_df.iloc[cluster]
    return cluster_df, img_url