import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here
import re

parser = argparse.ArgumentParser(
    description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))


def average_sim(filename):
    """" Calculates the average cosine similarity between two topics crude and grain. """
    df = pd.DataFrame(pd.read_csv(filename, index_col=0))
    crude = df.filter(like='crude', axis=0)
    grain = df.filter(like='grain', axis=0)
    crude_crude = round(np.mean(cosine_similarity(crude, crude)), 2)
    grain_grain = round(np.mean(cosine_similarity(grain, grain)), 2)
    crude_grain = round(np.mean(cosine_similarity(crude, grain)), 2)
    grain_crude = round(np.mean(cosine_similarity(grain, crude)), 2)
    results = [crude_crude, grain_grain, crude_grain, grain_crude]
    return results


def print_table(results):
    print("Average similarity between {} {}.".format(
        "crude-crude", results[0]))
    print("Average similarity between {} {}.".format(
        "grain-grain", results[1]))
    print("Average similarity between {} {}.".format(
        "crude-grain", results[2]))
    print("Average similarity between {} {}.".format(
        "grain-crude", results[3]))


print_table(average_sim(args.vectorfile))
