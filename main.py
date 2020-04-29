from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def get_only_text(url):
    """ 
    return the title and the text of the article
    at the specified url
    """
    page = urlopen(url)
    soup = BeautifulSoup(page, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
  
    print ("=====================")
    print (text)
    print ("=====================")
 
    return soup.title.text, text    
 
     
url="ANY"
#text = get_only_text(url)


import nltk
#nltk.download('punkt')
sentences = []
for s in text:
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x]
#sentences[30:40]

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append
    
    
    
sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
# Extract top 15 sentences as the summary representation
for i in range(10):
    print(ranked_sentences[i][1])





