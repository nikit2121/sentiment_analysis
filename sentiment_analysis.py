import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
nltk.download()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('/home/nikit/Desktop/Sentiment_analysis_on_movie_reviews/bag_of_popcorns/labeledTrainData.tsv',header = 0,\
                    delimiter="\t", quoting=3)
test = pd.read_csv('/home/nikit/Desktop/Sentiment_analysis_on_movie_reviews/bag_of_popcorns/testData.tsv',header = 0,\
                    delimiter="\t", quoting=3)
print(train.shape)
clean_review = []
def process_train_data(review):
    # Remove HTML tags
    instance = BeautifulSoup(review)
    sentence_with_no_tags = instance.get_text()

    # Remove whats not a letter a-z A-Z (non-letter) and lower case all words
    sentence_with_no_tags = re.sub("[^a-zA-Z]"," ",sentence_with_no_tags)
    lower_case_words = sentence_with_no_tags.lower()
    # Split each word
    lower_case_words = lower_case_words.split()

    # Get all meaningful_words
    meaningful_words = [w for w in lower_case_words if not w in stopwords.words("english")]

    # join meaningful_words array as a sentence
    return(" ".join(meaningful_words))

# Cleaning and parsing all reviews
for i in range(0,2000):
    if i%200==0:
        print("Cleaning review %d of %d" %(i,2000))
    clean_review.append(process_train_data(train['review'][i]))

# Initialize the CountVectorizer object

vectorizer = CountVectorizer(analyzer='word', tokenizer=None,preprocessor=None,stop_words=None, \
                             max_features=5000)
train_data_features = vectorizer.fit_transform(clean_review)
train_data_features = train_data_features.toarray()

print "Training Random Forest"

forest = RandomForestClassifier(n_estimators=50)
forest = forest.fit(train_data_features,train['sentiment'][0:2000])


clean_test_review = []

for i in range(0,100):
    if i%10==0:
        print("Cleaning review %d of %d" %(i,100))
    clean_test_review.append(test['review'][i])

test_data_features = vectorizer.transform(clean_test_review)
test_data_features = test_data_features.toarray()


result = forest.predict(test_data_features)

np.savetxt('/home/nikit/Desktop/Sentiment_analysis_on_movie_reviews/bag_of_popcorns/prediction.csv',result)
