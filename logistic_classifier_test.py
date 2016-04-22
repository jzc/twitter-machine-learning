from twitter_machine_learning import load_tweets, TweetFeaturizer, create_train_set, cross_validate
import json

train_file = r".\train_set_1.json"
result_file = r".\logistic_results.json"

out = open(result_file, "w", encoding="utf8")

print("Loading tweets")
tweets = load_tweets(train_file)
print("Done")

for i in range(1,6):
    for j in range(1,11):
        print("Creating featurizer")
        tf = TweetFeaturizer(tweets, i, j)
        print("Done")
        
        print("Creating training set")
        train_set = create_train_set(tf, tweets)
        print("Done")
        
        results = cross_validate("L", train_set, 10)
        results["features"] = len(tf.all_ngrams)
        results["ngrams"] = i
        results["removed"] = j
        print("Number of features: {}".format(results["features"]))
        print("ngrams: {}".format(i))
        print("Removed: {}".format(j))
        print("Accuracy: {}".format(results["average"]["accuracy"]))
        print("Precision: {}".format(results["average"]["precision"]))
        print("Recall: {}".format(results["average"]["recall"]))    
        out.write(json.JSONEncoder().encode(results) + "\n")    