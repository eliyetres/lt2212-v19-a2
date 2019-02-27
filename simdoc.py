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
    print(args.vectorfile)
    df= pd.DataFrame(pd.read_csv(text, index_col=0))
    crude = df.filter(like='crude',axis=0)
    grain = df.filter(like='grain',axis=0)
    crude_crude = round(np.mean(cosine_similarity(crude, crude)), 2)
    grain_grain = round(np.mean(cosine_similarity(grain, grain)), 2)
    crude_grain = round(np.mean(cosine_similarity(crude,grain)), 2)
    grain_crude = round(np.mean(cosine_similarity(grain,crude)),2)
    print("Average similarity between {} {}.".format("crude-crude", crude_crude))
    print("Average similarity between {} {}.".format("grain-grain", grain_grain))
    print("Average similarity between {} {}.".format("crude-grain", crude_grain))
    print("Average similarity between {} {}.".format("grain-crude", grain_crude))
    



        
average_sim(args.vectorfile)

def print_table(results):
    """ Prints the results to the console """
    res = 0
    return res
