import string
import os
import time
import cPickle as pickle
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import PCA
from scipy.spatial import distance


def tokenize(text):
	text = text.lower() # lower case
	for e in set(string.punctuation+'\n'+'\t'): # remove punctuation and line breaks/tabs
		text = text.replace(e, ' ')	
	for i in range(0,10):	# remove double spaces
		text = text.replace('  ', ' ')
	#text = text.translate(string.punctuation)  # punctuation
	tokens = nltk.word_tokenize(text)
	text = [w for w in tokens if not w in stopwords.words('english')] # stopwords
	stems = []
	for item in tokens: # stem
		stems.append(PorterStemmer().stem(item))
	return stems


#path = '/Users/gene/Learn/nlp_test/Reuters21578-Apte-90Cat/training/gold/0001021'
#path = '/Users/gene/Learn/nlp_test/Reuters21578-Apte-90Cat/training/'

path = '/Users/gene/Learn/nlp_test/20_newsgroups'

#print [os.path.join(path, f) for f in os.listdir(path)]

token_dict = {}
groups = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
for g, group in enumerate(groups):
	print ("doing group %d / %d"%(g, len(groups)))
	posts = [p for p in os.listdir(os.path.join(path, group)) if os.path.isfile(os.path.join(path, group, p))]
	for post in posts:
		post_path = os.path.join(path, group, post)
		with open (post_path, "r") as p:
			raw_text = p.read()
			token_dict[post_path] = re.sub(r'[^\x00-\x7f]',r'', raw_text) 
	

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
tfs_reduced = TruncatedSVD(n_components=500, random_state=0).fit_transform(tfs)


#TRUNC SVD = LSA

#activations = np.array(data["activations"])
#pca = PCA(n_components=num_components)
#pca.fit(activations)
#data["pca_acts"] = pca.transform(activations)

docs = token_dict.keys()


def test():
	idx = int(len(docs) * random.random())
	query_doc = docs[idx]
	query_vec = tfs_reduced[idx]
	distances = []
	for v, vec in enumerate(tfs_reduced):
	    dst = distance.euclidean(query_vec, vec)
	    distances.append(dst)
	idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:]
	return_doc = docs[idx_closest[0]]
	print("QUERY\n==================\n%s\n\n%s\n\n===========\nRESULT\n%s\n===========\n\n%s\n========"%(query_doc, token_dict[query_doc], return_doc, token_dict[return_doc]))



	




lookups = []
for i, p1 in enumerate(data["pca_acts"]):
    distances = []
    for j, p2 in enumerate(data["pca_acts"]):
        dst = distance.euclidean(p1, p2)
        distances.append(dst)
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_closest]
    lookups.append(idx_closest)

json_data = []
for i,lookup in enumerate(lookups):
    json_data.append({"path":data["paths"][i], "lookup":lookup})

with open(output_path, 'w') as outfile:
    json.dump(json_data, outfile)




//////////////////////////////

PCA
 - simple example
 - eigenfaces
 - manifolds
---------
text retrieval
 x loading corpus, investigating
 x tokenize, remove stop words
 x tf-idf
 x PCA/SVD
 - custom query
 - kNN retrieval
--------
topic modeling
 x LSA/LDA
 - t-SNE (to ml4a-ofx)
---------
word2vec
 - w/ neural net
 - analogies
 - t-SNE
---------
NLP applications of word2vec
 - word2vec NN classifier
 - sentiment analysis
 - named entity recognition
 - POS
---------
skip-thoughts





other ideas
 - follow-up on text docs with wikipedia example





















