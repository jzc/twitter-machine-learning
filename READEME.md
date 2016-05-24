# twitter-machine-learning

https://medium.com/@justinzcai/using-machine-learning-to-analyze-twitter-for-real-time-influenza-surveillance-585981462eac

Required libraries: NLTK, scikit-learn, NumPy, SciPy, matplotlib
- **classifier_test.py** - Used to create scores for different classifiers pass option -c with either "N", "L", or "S" (naive bayes, logistic, or SVC)
- **classify_new_data.py** - Use this script to classify new/unseen data. Data is read from `new_dir` (`.\flu_related_tweets`) and outputed in `classified_dir` (`.\classifed_tweets`)
- **compare_offical_classifed.py** - Compares the offical data to the classified data. Data is read from `classified_dir` (`.\classified_tweets`) and `offical_file` (`.\offical_data.json`)
- **find_flu_related_tweets.py** - Use this script to find tweets from a directory that are flu related. Assumes the files are gzipped. Run in command line with option -i <filename>.
- **label_more_training.py** - Experimental. Although never use in the project, it's original use was to expand the amount of relevant tweets in the training set.
- **twitter_machine_learning.py** - Main module that is used in most other scripts, handles implmentation of classifers
- **visualize_heat.py** - Makes a heatmap visulaztion of the classifier scores.