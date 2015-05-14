import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score

def load_file():
    inpTweets = csv.reader(open('/home/saurabh/Desktop/save/my_coding_algo/apple.csv', 'rb'), delimiter=',', quotechar='|')
    with open('/home/saurabh/Desktop/save/my_coding_algo/apple.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=',',quotechar='|')
        reader.next()
        data =[]
        target = []
        for row in reader:
		try:
			tweet = row[4]    
			senti = row[1]
			 
		except IndexError:
			tweet = 'null'
			seti = 'null'

		data.append(tweet)
		target.append(senti)
	return data,target

# preprocess creates the term frequency matrix for the review data set
def preprocess():
    data,target = load_file()
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)

    return tfidf_data

def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.4,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict((data_test))
    evaluate_model(target_test,predicted)


def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))

def main():
    data,target = load_file()
    tf_idf = preprocess()
    learn_model(tf_idf,target)

main()
