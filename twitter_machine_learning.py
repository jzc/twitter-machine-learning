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

def cross_validate(classifier, train_set, num_folds):
    subset_size = int(len(train_set)/num_folds)
    results = {"folds":[], "average":{}}
    for i in range(num_folds):
        print("fold " + repr(i))
        test_fold = train_set[i*subset_size:][:subset_size]
        train_fold = train_set[:i*subset_size] + train_set[(i+1)*subset_size:]
        if classifier == "N":
            name = "NaiveBayes"
            classifier = create_naive_bayes_classifier(train_fold)
        elif classifier == "L":
            name = "LogisticRegression"
            classifier = create_logistic_classifier(train_fold)
        elif classifier == "S":
            name = "SVC"
            classifier = create_svc_classifier(train_fold)
        test_fold_classified = classifier.classify_many(d for (d, l) in test_fold)
        test_fold_actual = [l for (d,l) in test_fold]
        accuracy, precision, recall = scores(test_fold_actual, test_fold_classified)
        results["folds"].append({"accuracy":accuracy,"precision":precision,"recall":recall})
    results["average"]["accuracy"] = sum([f['accuracy']for f in results['folds']])/num_folds
    results["average"]["precision"] = sum([f['precision']for f in results['folds']])/num_folds
    results["average"]["recall"] = sum([f['recall']for f in results['folds']])/num_folds
    return results
    
def create_train_set(tf, tweets):
    return [(tf.featurize(tweet[0]), tweet[1]) for tweet in tweets]

def load_tweets(file_name):
    #Load tweets
    file = open(file_name, "r", encoding="utf8")
    tweets = []
    for line in file:
        parsed = json.loads(line)
        tweets.append((parsed["text"], "REL" if parsed["type"] in ("REL", "WSH", "GSH") else "IRR"))
    return tweets
    
def create_naive_bayes_classifier(train_set):
    return nltk.classify.NaiveBayesClassifier.train(train_set)
    
def create_logistic_classifier(train_set):
    return nltk.classify.SklearnClassifier(LogisticRegression()).train(train_set)
    
def create_svc_classifier(train_set):    
    return nltk.classify.SklearnClassifier(SVC(class_weight='balanced')).train(train_set)