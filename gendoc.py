import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer


# add whatever additional imports you may need here
import re
import string
import nltk
from nltk.probability import FreqDist
import operator

# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!
fpath = '/reuters-topics/'


# FUNCTIONS

def get_vocab(fpath, top_m=None):
    """  Set -B

    Opens and processes every file in reuters+subfolders 
    The top-m most frequent word by count will be used, the rest will be filtered out. If it isn't set, all words will be used. (This is a failsafe in case the  documents doesnt fit in memory.) """
    vocab = ""
    for dirpath, _, files in os.walk(os.getcwd()+fpath):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            with open(fname) as myfile:
                r = myfile.read().lower()
                r = re.sub(r'([^\w\s]|[\n\'])', '', r)
                vocab += r  # add words to corpus
    fdist = FreqDist(vocab.split())
    sorted_d = list(
        sorted(fdist.items(), key=operator.itemgetter(1), reverse=True))
    if top_m is not None:
        vocab = sorted_d[:top_m]
    else:
        vocab = sorted_d
    return vocab


def getarticles(fpath, vocab, foldername):
    """ Opens and preprocesses files from subfolder.
    Returns a dict with article name and words and their counts compared to the total word count """
    dircorpus = {}
    # for every article
    for filename in os.listdir(os.getcwd()+fpath+foldername):
        articlecorpus = {}
        text = ""
        with open(os.getcwd()+fpath+foldername+"/"+filename) as myfile:
            r = myfile.read().lower()
            r = re.sub(r'([^\w\s\']|\n)', '', r)
            text += r  # add words to corpus
            text = text.split()
            for word in vocab:
                articlecorpus[word[0]] = 0
            for word in text:
                if word in dict(vocab).keys():
                    articlecorpus[word] += 1
        articlecorpus = dict(sorted(articlecorpus.items(),
                                    key=operator.itemgetter(1), reverse=True))
        # Add all subdicts to a nested dict
        dircorpus[filename] = articlecorpus
    return dircorpus



def create_rawdata(data_dict):
    """ Creates a Pandas data object from a nested dict of articles and their words and counts """
    raw_data = pd.DataFrame.from_dict(data_dict, orient="index")
    dropped_lines = raw_data[raw_data.duplicated()].index.tolist()
    print("Dropped {} files: ".format((len(dropped_lines))))
    print("---------------------\n", dropped_lines)
    raw_data = raw_data.drop_duplicates()
    # print(datafr)
    return raw_data


def create_tf_idf(raw_data):
    """ Set -T 

    Transforms raw data into TF-IDF values (after filtering -B)."""
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(raw_data)
    td_if_data = tfidf.toarray()
    # print(td_if_data)
    return td_if_data


def create_svd(dataframe, n):
    """ Set -S 

    Transform the term-document matrix by klearn's TruncatedSVD  operation into a document matrix with a feature space of dimensionality n. """
    svd = TruncatedSVD(n)
    dataframe = dataframe.values
    svd_data = svd.fit_transform(dataframe)
    # print(svd_data)
    return svd_data


def print_to_file(data, filename):
    """Takes a data object and prints it to a .csv file. """
    data.to_csv(filename+".csv")
    #data.to_csv("pandas2.csv", index=False) # Without article names

vocab = get_vocab(fpath, 50)
crude = getarticles(fpath, vocab, "crude")
grain = getarticles(fpath, vocab, "grain")
all={"crude":crude, "grain":grain}

raw_data = create_rawdata(all)



print(all)


print(create_svd(raw_data, 2)) # 2 dimensional data

print_to_file(raw_data, "outtest")

''' import json
with open('crude.txt', 'w') as file:
     file.write((crudedf)) # use `json.loads` to do the reverse '''


parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true",
                    help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()

print("Loading data from directory {}.".format(args.foldername))

if not args.basedims:
    print("Using full vocabulary.")
    vocab = get_vocab(fpath)

else:
    print("Using only top {} terms by raw count.".format(args.basedims))
    vocab = get_vocab(fpath, args.basedims)

if args.tfidf:
    print("Applying tf-idf to raw counts.")
    create_tf_idf(crude)
    create_tf_idf(grain)

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(
        args.svddims))

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.

print("Writing matrix to {}.".format(args.outputfile))
