# coding: utf-8

from collections import OrderedDict
from pymongo import MongoClient
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from sklearn import metrics

import json
import pickle
import rumor_terms
import re
import numpy
import random
import os

# db info
client = MongoClient('z') #fix this

#vectorizor info
analyzer = 'char_wb'
ngram_range = (1,5)
stopwords = 'english'
tfidf = False

class InvalidClassifierError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# extract the text column from a DataFrame
class textExtractor(TransformerMixin):
    def transform(self, X, **transform_params):
        return X['text'].values
    def fit(self, X, y=None, **fit_params):
        return self

# extract the features column from a DataFrame
class booleanFeatureExtractor(TransformerMixin):
    def transform(self, X, **transform_params):
        return X['features'].apply(lambda x: json.loads(x)).values
    def fit(self, X, y=None, **fit_params):
        return self

# classifier only based on keyword search for baseline statistics
class KeyWord(object):
    def __init__(self,keywords):
        self.keywords = keywords
    def predict(self,test_data):
        results = []
        for text in test_data['text'].values:
            match = 0
            for keyword in self.keywords:
                if keyword in text:
                    match = 1
            results.append(match)
        return results

# revomve rumore and event specific stopwords
# update this to use nltk/scikit-learn?
def remove_stopwords(words,event,rumor):
    stop_words = []
    if rumor:
        stop_words += rumor_terms.filter_words[rumor]
    if event:
        stop_words += rumor_terms.event_terms[event]
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

def find_mention(text):
    mentions = [
        ur'\u201c' + '@.*?:',
        '"@.*?:',
        'via @.*?:',
        'via @.*?\b',
        '@.*?\b',
    ]
    for x in mentions:
        if re.match(x,text):
            return True
    return False

# wrapper for scrubbing entire tweet
def process_tweet(tweet,event=None,rumor=None):
    text = scrub_tweet(tweet['text'])
    words = re.findall(r"[\w']+", text)
    if event or rumor:
        words = remove_stopwords(words,event,rumor)
    cleaned = ''
    for word in words:
        cleaned += word + ' '
    return cleaned

def get_tweet_meta_data(tweet,event,rumor):
    unique_id = int(tweet['db_id'])
    if event == 'sydeysiege':
        mapping = client['sydeysiege_cache'][rumor].find_one({'db_id':unique_id})
    else:
        mapping = client['rumor_compression'][rumor].find_one({'db_id':unique_id})
    if mapping:
        tweet_id = mapping['id'][0]
        full_tweet = client[event][rumor].find_one({'id':tweet_id})
        return full_tweet
    else:
        return None

# import all data from mongo into a dataframe with columns text, class, and
# rumor
# pos = 1, neg = 0
def import_training_data(fname=None,verbose=False):
    count = 0
    result = DataFrame({'text':[],'class':[],'rumor':[],'event':[],'features':[]})
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
                    #full_tweet = get_tweet_meta_data(tweet,event,rumor)
                    features = {}
                    #if full_tweet:
                    #    features['has_mention'] = find_mention(full_tweet['text'])
                    #else:
                    #    features['has_mention'] = False
                    if '?' in tweet['text']:
                        features['is_question'] = True
                    else:
                        features['is_question'] = False
                    text = process_tweet(tweet,event,rumor)
                    if "Uncertainty" in tweet['second_final']:
                        classification = 1
                    else:
                        classification = 0
                    result = result.append(DataFrame({
                        'text':text,
                        'class':classification,
                        'rumor':rumor,
                        'event':event,
                        'features':json.dumps(features)
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

def format_data_for_uncertainty_classification(event,fname=None,verbose=False):
    count = 0
    result = DataFrame({'text':[],'class':[],'rumor':[],'event':[],'features':[]})
    if verbose:
        print 'processing data from %s, %s' % (event)
    examples = client[event]['tweets'].find()
    for tweet in examples:
        if tweet['text']:
            #full_tweet = get_tweet_meta_data(tweet,event,rumor)
            features = {}
            #if full_tweet:
            #    features['has_mention'] = find_mention(full_tweet['text'])
            #else:
            #    features['has_mention'] = False
            if '?' in tweet['text']:
                features['is_question'] = True
            else:
                features['is_question'] = False
            text = process_tweet(tweet,event=event)
            if "Uncertainty" in tweet['second_final']:
                classification = 1
            else:
                classification = 0
            result = result.append(DataFrame({
                'text':text,
                'class':classification,
                'rumor':rumor,
                'event':event,
                'features':json.dumps(features)
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


# standard kfold validation
def kfold_split(labled_data,n_folds,fname=None):
    result = KFold(n=len(labled_data), n_folds=n_folds)
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    return result

# kfold validation where each "fold" is a rumor
def rumor_split(labled_data,fname=None):
    test_indices = []
    train_indices = []
    for event in rumor_terms.event_rumor_map:
        for rumor in rumor_terms.event_rumor_map[event]:
            test_temp = labled_data.loc[labled_data['rumor'] == rumor].index.tolist()
            train_temp = labled_data.loc[labled_data['rumor'] != rumor].index.tolist()
            test_indices.append(test_temp)
            train_indices.append(train_temp)
    result = zip(train_indices,test_indices)
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    return result

# kfold validation where each "fold" is an event
def event_split(labled_data,fname=None):
    test_indices = []
    train_indices = []
    for event in rumor_terms.event_rumor_map:
        test_temp = labled_data.loc[labled_data['event'] == event].index.tolist()
        train_temp = labled_data.loc[labled_data['event'] != event].index.tolist()
        test_indices.append(test_temp)
        train_indices.append(train_temp)
    result = zip(train_indices,test_indices)
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    return result

# unpickle datasets, or train/test index list
def unpickle_from_dicts(fname):
    fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
    f = open(fpath, 'r')
    unpickled = pickle.load(f)
    return unpickled


# validate the classifier over zipped training and testing datasets
# can be a single train/test pair or multiple zipped together
def validate_cl(labled_data,train_and_test,cl_type,verbose=False,split_type=None,fname=None,weighted=True):
    n = len(labled_data)
    scores = OrderedDict()
    scores['f1'] = []
    scores['recall'] = []
    scores['precision'] = []
    confusion = numpy.array([[0, 0], [0, 0]])
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'results/') + fname
        f = open(fpath, 'w')
        f.write('"rumor/fold","f1","recall","precision"\n')
    i = 0
    for x,y in train_and_test:
        if split_type == 'rumor' or split_type == 'event':
            train_data = labled_data.loc[x]

            test_data = labled_data.loc[y]
            test_lables = labled_data.loc[y]['class'].values
        else:
            train_data = labled_data.iloc[x]

            test_data = labled_data.iloc[y]
            test_lables = labled_data.iloc[y]['class'].values

        if cl_type == 'keyword':
            cl = KeyWord(rumor_terms.stemmed)
        else:
            cl = train_cl(train_data,cl_type,idf=False)

        predictions = cl.predict(test_data)
        print predictions
        confusion += metrics.confusion_matrix(test_lables, predictions)
        f1_score = metrics.f1_score(test_lables, predictions, pos_label=1)
        recall = metrics.recall_score(test_lables, predictions, pos_label=1)
        precision = metrics.precision_score(test_lables,predictions,pos_label=1)
        if split_type == 'rumor':
            rumor = labled_data.loc[y[0]]['rumor']
        elif split_type == 'event':
            rumor = labled_data.loc[y[0]]['event']
        else:
            rumor = str(i)
        if verbose:
            print rumor
            print 'tweets classified:', len(y)
            print 'f1: %s' % f1_score
            print 'recall: %s' % recall
            print 'precision: %s\n' % precision
        if fname:
            f.write('"%s","%s","%s","%s"\n' % (rumor,
                                               f1_score,
                                               recall,
                                               precision))
        if weighted:
            scores['f1'].append(f1_score * len(y))
            scores['recall'].append(recall * len(y))
            scores['precision'].append(precision * len(y))
        else:
            scores['f1'].append(f1_score)
            scores['recall'].append(recall)
            scores['precision'].append(precision)
        i += 1

    print 'Total tweets classified:', len(labled_data)
    for score in scores:
        if weighted:
            scores[score] = sum(scores[score])/n
        else:
            scores[score] = sum(scores[score])/i
    if fname:
        f.write('"%s","%s","%s","%s"\n' % ('total',
                                           scores['f1'],
                                           scores['recall'],
                                           scores['precision']))
    for score in scores:
        print '%s: %s' % (score,scores[score])
    print('Confusion matrix:')
    print(confusion)

# use a split train/test dataset to find all documents in test set labled as
# uncertainty
def find_uncertainty(labled_data,train_and_test,cl_type,fname,verbose=False):
    fpath = os.path.join(os.path.dirname(__file__),os.pardir,'results/') + fname
    f = open(fpath, 'w')
    f.write('"event","text"\n')
    for x,y in train_and_test:
        event = labled_data.loc[y[0]]['event']
        train_data = labled_data.loc[x]
        test_data = format_data_for_uncertainty_classification(event=event,
                                                               fname=None,
                                                               verbose=False)
        cl = train_cl(train_data,cl_type)
        print test_data
        predictions = cl.predict(test_data)
        print predictions
        pos_lables = test_data.iloc[numpy.where(predictions == 1)[0]]['text'].values
        print pos_lables

        if verbose:
            print rumor
            print 'tweets classified:', len(y)
        for text in pos_lables:
            f.write('"%s","%s"\n' % (rumor,
                                     text))

    print 'Total tweets classified:', len(labled_data)

# make a featureset and train a classifier
def train_cl(labled_data,cl_type,examples=None,fname=None):
    if cl_type == 'max_ent':
        cl = LogisticRegression()
    elif cl_type == 'nb':
        cl = MultinomialNB()
    elif cl_type == 'svm':
        cl = SVC()
    else:
        raise InvalidClassifierError('Not a valid classifier name')
    pipeline = Pipeline([
        ('features',FeatureUnion([
            ('bag_of_words',Pipeline([
                ('extractor',textExtractor()),
                ('vectorizer',CountVectorizer(analyzer='char_wb',
                                              ngram_range=(1,5),
                                              stop_words=None)),
                ('transformer',TfidfTransformer(use_idf=False))
            ])),
            ('uncertainty_terms',Pipeline([
                ('extractor',textExtractor()),
                ('vectorizer',CountVectorizer(analyzer='word',
                                              ngram_range=(1,1),
                                              stop_words=None,
                                              vocabulary=rumor_terms.stemmed)),
                ('transformer',TfidfTransformer(use_idf=False))
            ])),
            ('boolean_features',Pipeline([
                ('extractor',booleanFeatureExtractor()),
                ('vectorizer',DictVectorizer())
            ]))
        ])),
        ('classifier',cl)
    ])
    pipeline.fit(labled_data,
                 labled_data['class'].values)
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    if examples:
        print pipeline.predict(examples)
    return pipeline

def main():
    documents = import_training_data(verbose=True,fname='dataset_9-01.pickle')
    #documents = import_training_data(verbose=True)
    #documents = unpickle_from_dicts(fname='dataset_8-26.pickle')

    #counts = make_feature_set(labled_data=documents,verbose=True)

    #train_and_test = kfold_split(labled_data=documents,n_folds=10,fname='kfold_8-24.pickle')
    #train_and_test = kfold_split(labled_data=documents,n_folds=10)
    #train_and_test = rumor_split(labled_data=documents,fname='rumorfold_8-27.pickle')
    train_and_test = event_split(labled_data=documents,fname='eventfold_9-01.pickle')
    #train_and_test = unpickle_from_dicts(fname='rumorfold_8-26.pickle')

    '''validate_cl(labled_data=documents,
                train_and_test=train_and_test,
                cl_type='nb',
                verbose=True,
                split_type='rumor',
                #fname='max_ent_chargram_vocab_boolean_rumorfold_8-26.csv',
                weighted=False)'''

    find_uncertainty(labled_data=documents,
                     train_and_test=train_and_test,
                     cl_type='nb',
                     fname='uncertainty_tweets_9-01.csv',
                     verbose=True)

if __name__ == "__main__":
    main()
