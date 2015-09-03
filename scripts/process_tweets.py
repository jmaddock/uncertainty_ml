from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pandas import DataFrame

import pymongo
import rumor_terms
import numpy
import os
import pickle
import re
import json

# db info
client = MongoClient('z') #fix this
insert_db = 'uncertainty_ml_no_meta'

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

def pickle_from_db(event_list,fname,verbose=False):
    for event in event_list:
        result = DataFrame({'text':[],'event':[],'features':[],'unique_id':[]})
        count = 0
        if verbose:
            print 'processing data from %s' % (event)
        examples = client['uncertainty_ml'][event].find()
        for tweet in examples:
            if verbose and count % 1000 == 0 and count != 0:
                print 'processed %s tweets' % count
            if tweet['text']:
                result = result.append(DataFrame({
                    'text':tweet['text'],
                    'event':event,
                    'features':json.dumps(tweet['features']),
                    'unique_id':tweet['unique_id']
                },index=[count]))
                count += 1
                if count == 10000:
                    break
        result = result.reindex(numpy.random.permutation(result.index))

        fpath = os.path.join(os.path.dirname(__file__),os.pardir,'dicts/') + event + '_' + fname
        f = open(fpath, 'w')
        pickle.dump(result,f)
        f.close()
        if verbose:
            print result

def process_and_insert_meta_data(event_list,verbose=False):
    for event in event_list:
        try:
            unique_id = long(client[insert_db][event].find().sort('unique_id',-1).limit(1).next()['unique_id'] + 1)
        except StopIteration:
            unique_id = long(0)
        count = 0
        if verbose:
            print 'processing data from %s' % (event)
        examples = client[event]['tweets'].find()
        for tweet in examples:
            if unique_id == 1:
                if verbose:
                    print 'creating indexes'
                    client[insert_db][event].ensure_index('text')
                    client[insert_db][event].ensure_index('unique_id')
            if verbose and count % 1000 == 0 and count != 0:
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
                match = client[insert_db][event].find_one({'text':text})
                if match:
                    client[insert_db][event].update(
                        {'unique_id':match['unique_id']},
                        {
                            '$addToSet':{
                                'id':tweet['id'],
                                'entities.user_mentions':tweet['entities']['user_mentions'],
		                #'entities.symbols':tweet['entities']['symbols'],
		                #'entities.trends':tweet['entities']['trends'],
		                'entities.hashtags':tweet['entities']['hashtags'],
		                'entities.urls':tweet['entities']['urls'],
                                'created_ts':tweet['created_ts'],
			    }
                        }
                    )
                else:
                    client[insert_db][event].insert({
                        'unique_id':unique_id,
                        'id':[tweet['id']],
                        'entities':{
                            'user_mentions':[tweet['entities']['user_mentions']],
		            #'symbols':[tweet['entities']['symbols']],
		            #'trends':[tweet['entities']['trends']],
		            'hashtags':[tweet['entities']['hashtags']],
		            'urls':[tweet['entities']['urls']],
                        },
                        'created_ts':[tweet['created_ts']],
                        'text':text,
                        'event':event,
                        'features':features
                    })
                    unique_id += 1
                count += 1

def process_and_insert(event_list,verbose=False):
    for event in event_list:
        try:
            unique_id = long(client[insert_db][event].find().sort('unique_id',-1).limit(1).next()['unique_id'] + 1)
        except StopIteration:
            unique_id = long(0)
        count = 0
        if verbose:
            print 'processing data from %s' % (event)
        examples = client[event]['tweets'].find({'lang':'en'})
        for tweet in examples:
            if unique_id == 1:
                if verbose:
                    print 'creating indexes'
                    client[insert_db][event].create_index([('text',pymongo.DESCENDING)],unique=True)
                    client[insert_db][event].ensure_index('unique_id')
            if verbose and count % 1000 == 0 and count != 0:
                print 'processed %s tweets' % count
            if tweet['text']:
                features = {}
                if '?' in tweet['text']:
                    features['is_question'] = True
                else:
                    features['is_question'] = False
                text = process_tweet(tweet,event=event)
                document = {
                    'unique_id':unique_id,
                    'id':tweet['id'],
                    'text':text,
                    'event':event,
                    'features':features
                }
                try:
                    client[insert_db][event].insert(document)
                    unique_id += 1
                except DuplicateKeyError:
                    pass
            count += 1

def main():
    process_and_insert(event_list=['sydneysiege','new_boston','mh17'],
                       verbose=True
    )
    '''pickle_from_db(event_list=['sydneysiege'],
                   fname='testdump_9-02.pickle',
                   verbose=True)'''

if __name__ == "__main__":
    main()
