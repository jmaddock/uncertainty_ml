from pymongo import MongoClient
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.collocations import *
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import islice, izip
import nltk
import string
import os
import re
import rumor_terms
import random
import csv

class UncertaintyAnalysis(object):

    def __init__(self,stem,host=None,bigram=False):
        if host:
            self.client = MongoClient(host)
        else:
            self.client = MongoClient()
        self.bigram = bigram
        self.stem = stem
        self.uncertainty_total = 0
        self.baseline_total = 0
        self.top_words = Counter()
        self.baseline_top_words = Counter()
        self.results = {}

    def _remove_stopwords(self,words,event,rumor):
        stop_words = rumor_terms.filter_words[rumor] + rumor_terms.event_terms[event]
        if not self.bigram:
            stop_words += stopwords.words('english')
        filtered_words = [re.sub("'","",w.lower()) for w in words if not re.sub("'","",w.lower()) in stop_words]
        return filtered_words

    def _remove_bigram_stopwords(self,bigrams):
        filtered_words = []
        for w in bigrams:
            if (w[0] in stopwords.words('english')) and (w[1] in stopwords.words('english')):
                pass
            else:
                filtered_words.append(w)
        return filtered_words

    def _stem_words(self,words):
        stemmed = []
        for item in words:
            stemmed.append(PorterStemmer().stem(item))
        return stemmed

    def _scrub_tweet(self,text,scrub_url=True):
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

    def process_tweet(self,tweet,event,rumor,return_list=True):
        text = self._scrub_tweet(text=tweet['text'])
        words = re.findall(r"[\w']+", text)
        words = self._remove_stopwords(words,event,rumor)
        if self.stem:
            words = self._stem_words(words)
        if return_list:
            return words
        else:
            cleaned = ''
            for word in words:
                cleaned += word + ' '
            return cleaned

    def _find_bigrams(self,word_list):
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(word_list)
        finder.nbest(bigram_measures.likelihood_ratio, 10)

    def _count_all_words(self,tweet_list,event,rumor):
        word_count = 0
        for tweet in tweet_list:
            word_count += len(self.process_tweet(tweet,event,rumor))
        return word_count

    def top_uncertainty_words(self,event,rumor):
        db = self.client[event][rumor]
        uncertainty_tweet_list = [x for x in db.find({'codes.second_code':'Uncertainty'})]
        self.uncertainty_total += self._count_all_words(uncertainty_tweet_list,event,rumor)
        baseline_tweet_list = [x for x in db.find({'$and':[{'codes.first_code':{'$ne':'Unrelated'}},{'codes.first_code':{'$ne':'Uncodable'}}]})]

        self.baseline_total += self._count_all_words(baseline_tweet_list,event,rumor)
        print '[INFO] creating baseline counts'
        for tweet in baseline_tweet_list:
            #try:
                if self.bigram:
                    s = self.process_tweet(tweet=tweet)
                    filtered_words = self._remove_bigram_stopwords(zip(s, islice(s, 1, None)))
                else:
                    filtered_words = self.process_tweet(tweet,event,rumor)
                self.baseline_top_words.update(filtered_words)
            #except TypeError:
            #    print tweet['text']
            #    print 'error'
                pass

        print '[INFO] creating uncertainty counts'
        for tweet in uncertainty_tweet_list:
            #try:
                if self.bigram:
                    s = self.process_tweet(tweet=tweet)
                    filtered_words = self._remove_bigram_stopwords(zip(s, islice(s, 1, None)))
                else:
                    filtered_words = self.process_tweet(tweet,event,rumor)
                self.top_words.update(filtered_words)
            #except TypeError:
            #    print tweet['text']
            #    print 'error'

    def normalize_counts(self,output=True):
        print '[INFO] normalizing counts'
        for count in self.top_words:
            try:
                normalized = float(self.top_words[count])/float(self.uncertainty_total)
            except ZeroDivisionError:
                print 'error: %s' % self.top_words[count]
            normalized_base = float(self.baseline_top_words[count])/float(self.baseline_total)
            self.results[count] = normalized - normalized_base
        if output:
            print '[INFO] sorting'
            ordered_result = [x for x in self.results]
            ordered_result.sort(key=lambda x: self.results[x],reverse=True)
            for x in ordered_result[:50]:
                if self.bigram:
                    print x[0],x[1],self.results[x]
                else:
                    print x,self.results[x]

    def multiple_event_uncertainty_terms(self,output=True):
        for event in rumor_terms.compression_rumor_map:
            for rumor in rumor_terms.compression_rumor_map[event]:
                print event, rumor
                self.top_uncertainty_words(event,rumor)
        self.normalize_counts(output)

    def _read_uncertianty_terms(self,path):
        reader = fpath = os.path.join(os.path.dirname(__file__),os.pardir,'data/') + path
        with open(fpath, 'rb') as codesheet:
            reader = csv.reader(codesheet)
            first_row = True
            header = []
            terms = []
            for row in reader:
                if first_row:
                    header = row
                    first_row = False
                else:
                    terms.append(row[0].decode('latin-1').encode('utf-8'))
        return terms

    def search_uncertianty(self,term_list,num=1):
        for i in xrange(0,num):
            fname = 'uncertainty_sample_tweets_no_stem_filtered_%i.csv' % i
            f = utils.write_to_samples(path=fname)
            f.write('term,id,tweet text\n')
            for term in term_list:
                new_term = r'\b'+term+r'\b'
                query = {'text':re.compile(new_term,re.IGNORECASE)}
                tweets = self.db.find(query)
                tweet_list = [x for x in tweets]
                if len(tweet_list) < 100:
                    print 'threw out term: ' + term
                    print str(len(tweet_list)) + ' tweets'
                else:
                    result = []
                    print 'TERM: ' + term
                    while len(result) <= 10:
                        tweet = random.choice(tweet_list)
                        text = self._scrub_tweet(tweet['text'],scrub_url=True)
                        if len(result) == 0:
                            result.append(tweet)
                        else:
                            unique = True
                            for y in result:
                                if nltk.metrics.edit_distance(text,y['text']) < 10:
                                    unique = False
                            if unique:
                                result.append(tweet)
                    for x in result:
                        s = '"%s","%s","%s"\n' % (term,x['id'],x['text'])
                        f.write(s.encode('utf-8'))

    def find_rumors(self):
        term_list = self._read_uncertianty_terms(path='all_rumor_uncertainty_no_stem_filtered.csv')
        self.search_uncertianty(term_list=term_list,num=10)

    def find_random(self,num=1):
        for i in xrange(0,num):
            print i
            fname = 'random_sample_tweets_%i.csv' % i
            f = utils.write_to_samples(path=fname)
            f.write('term,id,tweet text\n')
            count = self.db.find().count()
            rand = random.randrange(count)
            tweet_list = self.db.find().skip(rand)
            result = []
            while len(result) <= 257:
                tweet = tweet_list.next()['text']
                text = self._scrub_tweet(tweet,scrub_url=True)
                if len(result) == 0:
                    result.append(tweet)
                else:
                    unique = True
                    for y in result:
                        if nltk.metrics.edit_distance(text,y) < 10:
                            unique = False
                    if unique:
                        result.append(text)
            for x in result:
                s = '"%s"\n' % (x)
                f.write(s.encode('utf-8'))

def compare_rumors(event_dict,bigram,stem):
    result_counter = Counter()
    for event in event_dict:
        print 'EVENT: %s' % event
        for rumor in event_dict[event]:
            print 'RUMOR: %s' % rumor
            u = UncertaintyAnalysis(event_name=event,rumor=rumor)
            result_counter.update(u.top_uncertainty_words(bigram=bigram,
                                                          stem=stem,
                                                          output=False,
                                                          sample_size=500))
    f = utils.write_to_data(path='10_rumor_uncertainty.csv')
    f.write('term,value\n')
    for x in result_counter.most_common(50):
        f.write('"%s",%f\n' % (x[0],x[1]))

def top_uncertainty(event_dict,bigram,stem):
    for event in event_dict:
        print 'EVENT: %s' % event
        for rumor in event_dict[event]:
            print 'RUMOR: %s' % rumor
            u = UncertaintyAnalysis(event_name=event,rumor=rumor)
            u.top_uncertainty_words(bigram=bigram,stem=stem)

def search(event_name,baseline_event_dict=None):
    if baseline_event_dict:
        compare_rumors(event_dict=baseline_event_dict,bigram=False,stem=False)
    u = UncertaintyAnalysis(event_name=event_name)
    u.find_rumors()

def main():
    # the event identifier
    event_dict = {
        'sydneysiege':['hadley','flag','lakemba','flag','suicide','airspace'],
        'mh17':['americans_onboard','american_falseflag'],
        'WestJet_Hijacking':['hijacking'],
        'power_outage':['foul_play'],
        'donetsk':['nuclear_detonation'],
        'baltimore':['purse']
    }
    # the rumor identifier
    u = UncertaintyAnalysis(host='z',stem=False)
    u.multiple_event_uncertainty_terms()
    #u.top_uncertainty_words(stem=False,bigram=True)
    #u.uncertainty_tf_idf()
    #top_uncertainty(event_dict=event_dict,bigram=True,stem=False)
    #search(event_name='baltimore',baseline_event_dict=event_dict)
    #search(event_name='baltimore')
    #u = UncertaintyAnalysis(event_name='baltimore')
    #u.find_random(num=4)

if __name__ == "__main__":
    main()
