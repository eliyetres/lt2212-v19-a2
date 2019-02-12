import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here
import re

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
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.tfidf:
    print("Applying tf-idf to raw counts.")

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(
        args.svddims))

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.

print("Writing matrix to {}.".format(args.outputfile))


# FUNCTIONS

def readfile(filename):
    """ Opens and reads a file """
    f = open(filename, "r")
    text = f.read()
    return text


def preprocess(text):
    """ Lowercases the text of a file and removes punctuation marks. """
    pp_text = []
    for line in text:
        line = line.lower()
        p = re.sub(r'[^\w\s]', '', line)
        pp_text.append(p)
    return pp_text


def topm(m):
    """ If -B is set,  only the top-m most frequent word by count will be used, the rest will be filtered out. If it isn't set, all words will be used. (This is a failsafe in case the  documents doesnt fit in memory.) """
    freqwords = 0
    return freqwords


def loaded_counts(filemame):
    """ If -T is set, the loaded counts (after filtering by the -B option) will be transformed from raw counts into tf-idf values. """
    # count
    # tf-idf values
    tfvalues = 0
    return tfvalues


def docmatrix(doc):
    """ If -S is set, then the term-document matrix will be transformed by klearn's TruncatedSVD  operation into a document matrix with a feature space of dimensionality n. """
    # document
    # matrix
    d_matrix = 0
    return d_matrix
