#import regex
import re
import csv
import pprint
import nltk.classify
import numpy as num

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector    
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


#Read the tweets one by one and process it
inpTweets = csv.reader(open('/home/yash/Desktop/saurabh/apple.csv', 'rb'), delimiter=',', quotechar='|')
stopWords = getStopWordList('/home/yash/Desktop/saurabh/final_destination/stopwords.txt')
count = 0;
featureList = []
tweets = []
for row in inpTweets:
    try:
    	sentiment = row[1]	#first ele is the sentiment pos or neg and other remiang is the tweet that we have to process further.
    	tweet = row[4]
	#print row[1],row[4]
    except IndexError:
	sentiment = 'null'	#first ele is the sentiment pos or neg and other remiang is the tweet that we have to process further.
    	tweet = 'null'
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

#testing 
testtweets = csv.reader(open('/home/yash/Desktop/saurabh/final_destination/test_cases/S5_tweets.csv', 'rb'), delimiter=',', quotechar='|')
# Test the classifierto
yash = []
i = 0
for row in testtweets:
	try:
		testTweet = row[3]
	except IndexError:	
		testTweet = 'null'
	processedTestTweet = processTweet(testTweet)
	sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
	print "testTweet = %s, sentiment = %s\n" % (testTweet, sentiment)
	#print sentiment
	yash.append(sentiment)
	

positive = yash.count(1)
negative = yash.count(-1)
neutral = yash.count(0)
irr = yash.count(2)
print '*************'
print yash
print "positive = %d, negative = %d, neutral= %d, irrelavent =%d\n" % (positive, negative, neutral, irr)
#print nltk.classify.accuracy(NBClassifier, training_set)
#NBClassifier.show_most_informative_features(100)
