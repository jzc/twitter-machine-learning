from os import listdir
from nltk.util import ngrams
from sklearn.linear_model import LogisticRegression
import json
import nltk
import string

def parse (text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text

class TweetFeaturizer(object):
    def __init__(self, tweets, n, remove):
        freq_dist = []
        self.all_ngrams = [] 
        for n in range(1,n+1):  
            for tweet in tweets:
                tweet = parse(tweet[0])
                tweet = tweet.split()
                tweet = set(ngrams(tweet, n))
                tweet = [' '.join(ngram) for ngram in tweet]    
                freq_dist.extend(tweet)
        freq_dist = nltk.FreqDist(freq_dist)
        legomena = [word for word in freq_dist if freq_dist[word] <= remove]
        list_all_ngrams = list(freq_dist.keys())
        for legomenon in legomena: list_all_ngrams.remove(legomenon)
        self.all_ngrams.extend(list_all_ngrams)
    def featurize(self, tweet):
        return ({ngram: (ngram in parse(tweet)) for ngram in self.all_ngrams })

test_dir = r".\flu_related_tweets"
train_file = r".\training_set.json"
n = 4
remove = 8

#Load tweets
file = open(train_file, "r", encoding="utf8")
tweets = []
for line in file:
    parsed = json.loads(line)
    tweets.append((parsed["text"], "REL" if parsed["type"] in ("WSH", "GSH") else "IRR"))
    
#Extract ngrams and remove hapaxes
tf = TweetFeaturizer(tweets, n, remove)
print("created ngrams")

#Create training set
train_set = [(tf.featurize(tweet[0]), tweet[1]) for tweet in tweets]
print("created training set")

#Create classifier
classifier = nltk.classify.SklearnClassifier(LogisticRegression())
classifier.train(train_set) 

for file in listdir(test_dir):
    print(r"{}\{}".format(test_dir, file))
    rel_count = 0
    irr_count = 0
    for line in open(r"{}\{}".format(test_dir, file), "r", encoding="utf8"):
        parsed = json.loads(line)
        if classifier.classify(tf.featurize(parsed["text"])) == "REL":
            rel_count = rel_count + 1
        else:
            irr_count = irr_count + 1
    print("REL: {} IRR: {}".format(rel_count, irr_count))