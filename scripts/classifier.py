# coding: utf-8

from pymongo import MongoClient
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from  sklearn import metrics

import pickle
import rumor_terms
import re
import numpy
import random

# db info
client = MongoClient() #fix this

#vectorizor info
analyzer = 'word'
ngram_range = (1,1)
stopwords = 'english'
tfidf = False

class InvalidClassifierError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# revomve rumore and event specific stopwords
# update this to use nltk/scikit-learn?
def remove_stopwords(words,event,rumor):
    stop_words = rumor_terms.filter_words[rumor] + rumor_terms.event_terms[event]
    filtered_words = [re.sub("'","",w.lower()) for w in words if not re.sub("'","",w.lower()) in stop_words]
    return filtered_words

# remove urls, hashtags, mentions
def scrub_tweet(text,scrub_url=True):
        temp = None
        s = ur'\u201c' + '@.*?:'
        while text is not temp:
            temp = text
            text = re.sub('RT .*?:','',text).strip()
            text = re.sub('"@.*?:','',text).strip()
            text = re.sub(s,'',text).strip()
            text = re.sub('via @.*?:','',text).strip()
            text = re.sub('via @.*?\b','',text).strip()
            text = re.sub('@.*?\b','',text).strip()
            if scrub_url is True:
                text = re.sub('http.*?\s|http.*?$','',text).strip()
        #print text
        return text

# wrapper for scrubbing entire tweet
def process_tweet(tweet,event,rumor):
    text = scrub_tweet(tweet['text'])
    words = re.findall(r"[\w']+", text)
    words = remove_stopwords(words,event,rumor)
    cleaned = ''
    for word in words:
        cleaned += word + ' '
    return cleaned

# import all data from mongo into a dataframe with columns text, class, and
# rumor
# pos = 1, neg = 0
def import_training_data(fname=None,verbose=False):
    count = 0
    result = DataFrame({'text':[],'class':[],'rumor':[],'event':[]})
    for event in rumor_terms.event_rumor_map:
        for rumor in rumor_terms.event_rumor_map[event]:
            if verbose:
                print 'processing data from %s, %s' % (event,rumor)
            pos_examples = [x for x in client['code_comparison'][rumor].find({'first_final':{'$in':['Affirm','Deny','Neutral']},'second_final':'Uncertainty'})]
            neg_examples = [x for x in client['code_comparison'][rumor].find({'first_final':{'$in':['Affirm','Deny','Neutral']},'second_final':{'$ne':'Uncertainty'}})]
            examples = pos_examples
            examples += random.sample(neg_examples,len(pos_examples))
            for tweet in examples:
                if tweet['text']:
                    text = process_tweet(tweet,event,rumor)
                    if "Uncertainty" in tweet['second_final']:
                        classification = 1
                    else:
                        classification = 0
                    result = result.append(DataFrame({
                        'text':text,
                        'class':classification,
                        'rumor':rumor,
                        'event':event
                    },index=[count]))
                    count += 1
    result = result.reindex(numpy.random.permutation(result.index))

    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    if verbose:
        print result
    return result

# DEPRECIATED -- included in pipeline
def make_feature_set(labled_data,fname=None,unpickle=False,verbose=False):
    if unpickle:
        labled_data = pickle.load(labled_data)
    if tfidf:
        count_vectorizer = TfidfVectorizer(analyzer=analyzer,
                                           ngram_range=ngram_range,
                                           stop_words=stopwords)
    else:
        count_vectorizer = CountVectorizer(analyzer=analyzer,
                                           ngram_range=ngram_range,
                                           stop_words=stopwords)
    print labled_data['text'].values
    counts = count_vectorizer.fit_transform(labled_data['text'].values)
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    if verbose:
        feature_names = count_vectorizer.get_feature_names()
        print counts
    return counts

# make a featureset and train a classifier
def train_cl(labled_data,cl_type,examples=None):
    if cl_type == 'max_ent':
        cl = LogisticRegression()
    elif cl_type == 'nb':
        cl = MultinomialNB()
    elif cl_type == 'svm':
        cl = SVC()
    else:
        raise InvalidClassifierError('Not a valid classifier name')
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer(analyzer=analyzer,
                                        ngram_range=ngram_range,
                                        stop_words=stopwords)),
        ('classifier',  cl)
    ])

    pipeline.fit(labled_data['text'].values,
                 labled_data['class'].values)
    if examples:
        print pipeline.predict(examples)
    return pipeline

# validate the classifier over zipped training and testing datasets
# can be a single train/test pair or multiple zipped together
def validate_cl(labled_data,train_and_test,verbose=False,by_rumor=False):
    n = len(labled_data)
    scores = {
        'f1':[],
        'recall':[],
        'precision':[],
    }
    confusion = numpy.array([[0, 0], [0, 0]])
    for x, y in train_and_test:
        train_data = labled_data.loc[x]

        test_text = labled_data.loc[y]['text'].values
        test_lables = labled_data.loc[y]['class'].values

        cl = train_cl(train_data,'nb')
        predictions = cl.predict(test_text)

        confusion += metrics.confusion_matrix(test_lables, predictions)
        f1_score = metrics.f1_score(test_lables, predictions, pos_label=1)
        recall = metrics.recall_score(test_lables, predictions, pos_label=1)
        precision = metrics.precision_score(test_lables,predictions,pos_label=1)
        if verbose:
            if by_rumor:
                print labled_data.loc[y[1]]['rumor']
            print'tweets classified:', len(y)
            print 'f1: %s' % f1_score
            print 'recall: %s' % recall
            print 'precision: %s\n' % precision
        scores['f1'].append(f1_score)
        scores['recall'].append(recall)
        scores['precision'].append(precision)

    print('Total tweets classified:', len(labled_data))
    for score in scores:
        print '%s: %s' % (score,sum(scores[score])/len(scores[score]))
    print('Confusion matrix:')
    print(confusion)

# standard kfold validation
def kfold_split(labled_data,n_folds):
    return KFold(n=len(labled_data), n_folds=n_folds)

# kfold validation where each "fold" is a rumor
def rumor_split(labled_data):
    test_indices = []
    train_indices = []
    for event in rumor_terms.event_rumor_map:
        for rumor in rumor_terms.event_rumor_map[event]:
            test_temp = labled_data.loc[labled_data['rumor'] == rumor].index.tolist()
            train_temp = labled_data.loc[labled_data['rumor'] != rumor].index.tolist()
            test_indices.append(test_temp)
            train_indices.append(train_temp)
    return zip(train_indices,test_indices)

# kfold validation where each "fold" is an event
def event_split(labled_data):
    test_indices = []
    train_indices = []
    for event in rumor_terms.event_rumor_map:
        test_temp = labled_data.loc[labled_data['event'] == event].index.tolist()
        train_temp = labled_data.loc[labled_data['event'] != event].index.tolist()
        test_indices.append(test_temp)
        train_indices.append(train_temp)
    return zip(train_indices,test_indices)

# DEPRECIATED
'''def train_and_validate_kfold(labled_data,n_folds=None,split_by_rumor=False):
    n = len(labled_data)
    scores = {
        'f1':[],
        'recall':[],
        'precision':[],
    }
    confusion = numpy.array([[0, 0], [0, 0]])
    if split_by_rumor:
        test_indices = []
        train_indices = []
        for event in rumor_terms.event_rumor_map:
            for rumor in rumor_terms.event_rumor_map[event]:
                test_temp = []
                train_temp = []
                for x in labled_data:
                    if x['rumor'] == rumor:
                        test_temp.append(x['index'])
                    else:
                        train_temp.append(x['index'])
                test_indices.append(test_temp)
                train_indices.append(train_temp)
        k_fold = zip(train_indices,test_indices)
    else:
        k_fold = ## finish this

    for train_indices, test_indices in k_fold:
        train_data = labled_data.iloc[train_indices]

        test_text = labled_data.iloc[test_indices]['text'].values
        test_lables = labled_data.iloc[test_indices]['class'].values

        cl = train_cl(train_data)
        predictions = cl.predict(test_text)

        confusion += metrics.confusion_matrix(test_lables, predictions)
        f1_score = metrics.f1_score(test_lables, predictions, pos_label=1)
        recall = metrics.recall_score(test_lables, predictions, pos_label=1)
        precision = metrics.precision_score(test_lables,predictions,pos_label=1)
        scores['f1'].append(f1_score)
        scores['recall'].append(recall)
        scores['precision'].append(precision)

    print('Total tweets classified:', len(labled_data))
    for score in scores:
        print '%s: %s' % (score,sum(scores[score])/len(scores[score]))
    print('Confusion matrix:')
    print(confusion)'''

def main():
    documents = import_training_data(verbose=True)
    #counts = make_feature_set(labled_data=documents,verbose=True)
    #train_and_test = kfold_split(labled_data=documents,n_folds=10)
    train_and_test = event_split(labled_data=documents,)
    validate_cl(labled_data=documents,train_and_test=train_and_test,verbose=True,by_rumor=True)

if __name__ == "__main__":
    main()
