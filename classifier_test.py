from twitter_machine_learning import TweetFeaturizer, cross_validate, create_train_set
import json

train_file = r".\training_set.json"
#out = open("classifier_results.txt", "w")
num_folds = 10

#Load tweets
file = open(train_file, "r", encoding="utf8")
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
        train_set = create_train_set(tf, tweets)
        print("created training set")
        subset_size = int(len(train_set)/num_folds)

        #Make classifier and cross validate
        for j in range(3):
            if j == 0:
                name = "NaiveBayes"
                results = cross_validate("N", train_set, 10)
            if j == 1:
                name = "LogisticRegression"
                results = cross_validate("L", train_set, 10)
            if j == 2:
                name = "SVC"
                results = cross_validate("S", train_set, 10)
            name = "{} ngrams: {} removed: {}".format(name, n, remove)
            for fold in results["folds"]:
                output = ', '.join((name,repr(fold["accuracy"]),repr(fold["precision"]),repr(fold["recall"])))
                print(output)    