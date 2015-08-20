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
        self.final_results = {}

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
            if tweet['text']:
                word_count += len(self.process_tweet(tweet,event,rumor))
        return word_count

    def top_uncertainty_words(self,event,rumor):
        db = self.client['code_comparison'][rumor]
        uncertainty_tweet_list = [x for x in db.find({'second_final':'Uncertainty'}) if x['text']]
        self.uncertainty_total += self._count_all_words(uncertainty_tweet_list,event,rumor)
        baseline_tweet_list = [x for x in db.find({'$and':[{'first_final':{'$ne':'Unrelated'}},{'codes.first_code':{'$ne':'Uncodable'}}]}) if x['text']]

        self.baseline_total += self._count_all_words(baseline_tweet_list,event,rumor)
        print '[INFO] creating baseline counts'
        for tweet in baseline_tweet_list:
                if self.bigram:
                    s = self.process_tweet(tweet=tweet)
                    filtered_words = self._remove_bigram_stopwords(zip(s, islice(s, 1, None)))
                else:
                    filtered_words = self.process_tweet(tweet,event,rumor)
                self.baseline_top_words.update(filtered_words)
            #except TypeError:
            #    print tweet['text']
            #    print 'error'

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
        results = {}
        for count in self.top_words:
            try:
                normalized = float(self.top_words[count])/float(self.uncertainty_total)
            except ZeroDivisionError:
                print 'error: %s' % self.top_words[count]
            normalized_base = float(self.baseline_top_words[count])/float(self.baseline_total)
            #print count, normalized, normalized_base, self.baseline_top_words[count]
            try:
                results[count] = normalized - normalized_base
            except ZeroDivisionError:
                print self.top_words['count']
        if output:
            print '[INFO] sorting'
            ordered_result = [x for x in results]
            ordered_result.sort(key=lambda x: results[x],reverse=True)
            for x in ordered_result[:50]:
                if self.bigram:
                    print x[0],x[1],results[x]
                else:
                    print x,results[x]
        return results

    def multiple_event_uncertainty_terms(self,output=True):
        final_keys = []
        for event in rumor_terms.event_rumor_map:
            for rumor in rumor_terms.event_rumor_map[event]:
                print event, rumor
                self.top_uncertainty_words(event,rumor)
        self.normalize_counts(output)

    def multiple_event_uncertainty_terms_normalized(self,output=True):
        final_keys = []
        normalized = []
        results = {}
        for event in rumor_terms.event_rumor_map:
            for rumor in rumor_terms.event_rumor_map[event]:
                print event, rumor
                self.top_uncertainty_words(event,rumor)
                counts = self.normalize_counts(output=False)
                normalized.append(counts)
                keys = counts.keys()
                if len(final_keys) == 0:
                    final_keys = keys
                else:
                    final_keys = list(set(final_keys).union(set(keys)))
                print len(keys)
                print len(final_keys)
                self.top_words = Counter()
                self.baseline_top_words = Counter()
                self.uncertainty_total = 0
                self.baseline_total = 0
        for term in final_keys:
            val = 1
            for x in normalized:
                if term in x:
                    val = val + x[term]
                results[term] = val
        if output:
            print '[INFO] sorting'
            ordered_result = [x for x in results]
            ordered_result.sort(key=lambda x: results[x],reverse=True)
            for x in ordered_result[:50]:
                if self.bigram:
                    print x[0],x[1],results[x]
                else:
                    print x,results[x]
        return results

def main():
    # the rumor identifier
    u = UncertaintyAnalysis(host='',stem=False)
    u.multiple_event_uncertainty_terms_normalized(output=True)
    #u.top_uncertainty_words(stem=False,bigram=True)
    #u.uncertainty_tf_idf()
    #top_uncertainty(event_dict=event_dict,bigram=True,stem=False)
    #search(event_name='baltimore',baseline_event_dict=event_dict)
    #search(event_name='baltimore')
    #u = UncertaintyAnalysis(event_name='baltimore')
    #u.find_random(num=4)

if __name__ == "__main__":
    main()
