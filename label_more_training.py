from twitter_machine_learning import *

test_dir = r".\flu_related_tweets"
train_file = r".\training_set.json"
n = 4
remove = 8

#Load tweets
tweets = load_tweets(train_file)

#Extract ngrams and remove hapaxes
tf = TweetFeaturizer(tweets, n, remove)
print("created featurizer")

#Create training set
train_set = create_train_set(tf, tweets)
print("created training set")

#Create classifier
classifier = nltk.classify.SklearnClassifier(LogisticRegression())
classifier.train(train_set) 

correct = 0
incorrect = 0
for file in listdir(test_dir):
    print(r"{}\{}".format(test_dir, file))
    rel_count = 0
    irr_count = 0
    for line in open(r"{}\{}".format(test_dir, file), "r", encoding="utf8"):
        parsed = json.loads(line)
        tweet = (parsed["text"], classifier.classify(tf.featurize(parsed["text"])))
        if  tweet[1] == "REL":
            if tweet not in tweets:     
                rel_count = rel_count + 1
                print("{}\n1:REL 2:IRR".format(parsed["text"]))
                n = ''
                while n not in ("1","2"):
                    n = input()
                if n == "1":
                    correct = correct + 1
                    tweets.append(tweet)
                else:
                    incorrect = incorrect + 1
        else:
            irr_count = irr_count + 1
    print("REL: {} IRR: {}".format(rel_count, irr_count))

file = open("{}.new".format(train_file), "w", encoding="utf8")
for (text, label) in tweets:
    file.write("{}\n".format(repr({"text":text, "type": label})))