from pymongo import MongoClient
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#import ml_config
import pickle
import rumor_terms
import re
import numpy

# db info
client = MongoClient('z')

#vectorizor info
analyzer = 'word'
ngram_range = (1,1)
stopwords = 'english'
tfidf = False

def remove_stopwords(words,rumor,event):
    stop_words = rumor_terms.filter_words[rumor] + rumor_terms.event_terms[event]
    filtered_words = [re.sub("'","",w.lower()) for w in words if not re.sub("'","",w.lower()) in stop_words]
    return filtered_words

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

def import_data(fname=None):
    result = DataFrame({'text':[],'class':[]})
    for event in rumor_terms.event_rumor_map:
        for rumor in rumor_terms.event_rumor_map[event]:
            rows = []
            index = []
            tweets = client[event][rumor].find({'codes.first_code':{'$in':['Affirm','Deny','Neutral']}})
            for tweet in tweets:
                text = scrub_tweet(tweet['text'])
                text = remove_stopwords(text,rumor,event)
                if "Uncertainty" in tweet['codes'][0]['second_code']:
                    classification = 1
                else:
                    classification = 0
                rows.append({'text':text,'class':classification})
                index.append(rumor)
            data = DataFrame(rows,index=index)
            result.append(data)
            result = result.reindex(numpy.random.permutation(data.index))
        break
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    return result

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
    counts = count_vectorizer.fit_transform(labled_data['text'].values)
    if fname:
        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
    if verbose:
        print count_vectorizer.vocabulary
    return counts

'''
def make_pipeline(labled_data):
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer()),
        ('classifier',  MultinomialNB()) ])

    pipeline.fit(data['text'].values, data['class'].values
                 pipeline.predict(examples) # ['spam', 'ham']
'''

def main():
    documents = import_data()
    make_feature_set(labled_data=documents,verbose=True)

if __name__ == "__main__":
    main()
