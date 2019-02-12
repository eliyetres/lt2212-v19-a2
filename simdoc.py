import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!
#hello
# add whatever additional imports you may need here
import re

parser = argparse.ArgumentParser(
    description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))



def average_sim(text):
    """" Prints four values, two for each topic.  
    
    For each of the two topics, print the average similarity of every document vector with that topic to every other vector with that same topic (for simplicity, including itself, if you want), averaged over the entire topic. the average similarity of every document vector with that topic to every document vector in the other topic, averaged over the entire topic. """


def print_table(results):
    """ Prints the results to the console """
    res = 0
    return res
