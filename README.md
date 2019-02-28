# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name:    Elin Hagman\
GitHub:     eliyetres\
GU:         gusihaliel

## Additional instructions

I created an error message when trying to trunkate to dimensions higher than the vocabulary count. The terminal prints the error message: "Error: Singular value decomposition dimensions must be lower than vocabulary limit." and exits without errors.

## Results and discussion

Running anything using the full vocabulary resulted in a MemoryError on my computer, so i ran the files in the server. All the output files are in the LT2212-vt-19-a2 folder on the server. 

gendoc.py outputs a csv file with article name as columns and words as rows. simdoc.py takes the output file and calculates the cosine similarity between the  articles from crude and grain respectively.

### Vocabulary restriction.

I restricted the vocabulary to 20 words. These words are the most frequent in all the articles and are be more significant.

### Result table


File names | Crude-Crude  | Grain-Grain | Crude-Grain | Grain-Crude 
--- | --- | --- | --- | --- 
allfiles|0.37|0.33|0.31|0.31
limit20|0.69|0.63|0.62|0.62
allfilestdif|0.11|0.10|0.07|0.07
limit20tdif|0.65|0.58|0.54|0.54
allfilessvd100|0.50|0.46|0.42|0.42
allfilessvd1000|0.37|0.33|0.31|0.31
allfilestdifsvsd00|0.26|0.22|0.18|0.18
allfilestdifsvd1000|0.11|0.10|0.07|0.07


### The hypothesis in your own words

### Discussion of trends in results in light of the hypothesis
