import os
import numpy as np

#Create a dictionary/map to store the word embeddings
embeddings_index = {}

#Load pre-computed word embeddings
#These can be dowloaded from https://nlp.stanford.edu/projects/glove/
#e.g., wget http://nlp.stanford.edu/data/glove.6B.zip
embeddings_size = "300"
f = open(os.path.join('.', 'class_02_embeddings\glove.6B.'+embeddings_size+'d.txt'), encoding='utf-8')

#Process file and load into structure
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#Compute distances among first X words (depending on your machine)
max_words = 30
from sklearn.metrics.pairwise import pairwise_distances
mat = pairwise_distances(list(embeddings_index.values())[:max_words])

#Replace self distances from 0 to inf (to use argmin)
np.fill_diagonal(mat, np.inf)

#Compute the most similar word for every word
min_0 = np.argmin(mat,axis=0)

#Save the pairs to a file
f_out = open('similarity_pairs_dim'+embeddings_size+'_first'+str(max_words)+'.txt','w')
for i,item in enumerate(list(embeddings_index.keys())[:max_words]):
    f_out.write(str(item)+' '+str(list(embeddings_index.keys())[min_0[i]])+'\n')

###
###Test the "king - man + woman = queen" analogy
###

#Compute embedding of the analogy
embedding_analogy = embeddings_index['king'] - embeddings_index['man'] + embeddings_index['queen']
#Find distances with the rest of the words
analogy_distances = np.empty(len(embeddings_index))
for i,item in enumerate(list(embeddings_index.values())):
    analogy_distances[i] = pairwise_distances(embedding_analogy.reshape(1, -1),item.reshape(1, -1))
#Print top 10 results
print([list(embeddings_index.keys())[i] for i in analogy_distances.argsort()[:10]])
print([list(embeddings_index.values())[i] for i in analogy_distances.argsort()[:10]])