import csv
import os
import re

from collections import Counter

fpath = r'../../../research_data/ml_uncertainty/uncertainty_terms_WestJet_Hijacking_hijacking_backup.csv'

def csv_tf(fpath):
    tf = Counter()
    with open(fpath, 'rb') as codesheet:
        reader = csv.reader(codesheet)
        headers = []
        for i,row in enumerate(reader):
            if i < 2:
                header = []
                for col in row:
                    if col == '':
                        header.append(header[:-1])
                    else:
                        header.append(col.lower())
                headers.append(header)
            else:
                for j,col in enumerate(row):
                    if headers[0][j] == 'text':
                        continue
                    elif headers[0][j] == 'uncertianty object':
                        continue
                    else:
                        text = re.sub("[\(\[].*?[\)\]]", "", col).strip().lower()
                        if text == 'x':
                            continue
                        elif text == '':
                            continue
                        else:
                            tf.update([text])
    terms_list = [term[0] for term in tf.most_common()]
    print terms_list

def term_validation(fpath):
    f = open(fpath,'r')
    results = [line[:-5] for line in f if line[-5:-3] == ' x']
    print results

if __name__ == "__main__":
    csv_tf(fpath)
    #term_validation(fpath)
