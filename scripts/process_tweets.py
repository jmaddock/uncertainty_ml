from pymongo import MongoClient
from pandas import DataFrame

import rumor_terms
import numpy
import os
import pickle
import re
import json

# db info
client = MongoClient('z') #fix this

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

def process_and_pickle(event_list,fname,insert=False,verbose=False):
    for event in event_list:
        count = 0
        result = DataFrame({'text':[],'event':[],'features':[]})
        if insert:
            insert_list = []
        if verbose:
            print 'processing data from %s' % (event)
        examples = client[event]['tweets'].find()
        for tweet in examples:
            if verbose and count % 1000 == 0:
                print 'processed %s tweets' % count
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
                result = result.append(DataFrame({
                    'text':text,
                    'event':event,
                    'features':json.dumps(features)
                },index=[count]))
                if insert:
                    insert_list.append({
                        'id':tweet['id'],
                        'text':text,
                        'event':event,
                        'features':features
                    })
                    if count % 1000 == 0:
                        client['uncertainty_ml'][event].insert(insert_list)
                        insert_list = []
                count += 1
        if insert:
            client['uncertainty_ml'][event].insert(insert_list)
            insert_list = []
        result = result.reindex(numpy.random.permutation(result.index))

        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + event + '_' + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
        f.close()
        if verbose:
            print result

def main():
    process_and_pickle(event_list=['sydneysiege','new_boston','mh17'],
                       fname='to_classify_9-02.pickle',
                       insert=True,
                       verbose=True
    )

if __name__ == "__main__":
    main()
