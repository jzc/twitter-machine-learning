from nltk.util import ngrams
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import nltk
import json
import string

def parse (text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text
    
def scores(reference, test):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for (r, t) in zip(reference, test):
        if r == "REL" and t == "REL":
            tp = tp + 1
        elif r == "IRR" and t == "IRR":
            tn = tn + 1
        elif r == "REL" and t == "IRR":
            fp = fp + 1
        elif r == "IRR" and t == "REL":
            fn = fn + 1
    print("tp: {} tn: {} fp: {} fn: {}".format(tp, tn, fp, fn))
    return ((tp + tn)/(tp + tn + fp + fn), tp/(tp + fp) if not tp + fp == 0 else None, tp/(tp + fn) if not tp + fn == 0 else None) 
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

file_name = r".\training_set.json"
out = open("classifier_results.txt", "w")
num_folds = 10
#Load tweets
file = open(file_name, "r", encoding="utf8")
tweets = []
for line in file:
    parsed = json.loads(line)
    tweets.append((parsed["text"], "REL" if parsed["type"] in ("WSH", "GSH") else "IRR"))
for n in range(1,6):
    for remove in range(1,11):     
        #Extract ngrams and remove hapaxes
        tf = TweetFeaturizer(tweets, n, remove)
        print("created ngrams")

        #Create training set
        train_set = [(tf.featurize(tweet[0]), tweet[1]) for tweet in tweets]
        print("created training set")
        subset_size = int(len(train_set)/num_folds)

        #Make classifier and cross validate
        for j in range(3):
            for i in range(num_folds):
                print("fold " + repr(i))
                test_fold = train_set[i*subset_size:][:subset_size]
                train_fold = train_set[:i*subset_size] + train_set[(i+1)*subset_size:]
                if j == 0:
                    name = "Naive Bayes"
                    classifier = nltk.classify.NaiveBayesClassifier.train(train_fold)
                elif j == 1:
                    name = "LogisticRegression"
                    classifier = nltk.classify.SklearnClassifier(LogisticRegression()).train(train_fold)
                elif j == 2:
                    name = "SVC"
                    classifier = nltk.classify.SklearnClassifier(SVC(class_weight='balanced')).train(train_fold)
                classifier.train(train_fold)
                test_fold_classified = classifier.classify_many(d for (d, l) in test_fold)
                test_fold_actual = [l for (d,l) in test_fold]
                accuracy, precision, recall = scores(test_fold_actual, test_fold_classified)
                name = "{} ngrams: {} removed: {}".format(name, n, remove)
                output = ', '.join((name,repr(accuracy),repr(precision),repr(recall)))
                print(output)
                out.write(output)
                out.write("\n")