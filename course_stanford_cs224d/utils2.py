import sys, os, re, json
import itertools
from collections import Counter
import time
from numpy import *

import pandas as pd






def load_wv_pandas(fname):
    return pd.read_hdf(fname, 'data')

def extract_wv(df):
    num_to_word = dict(enumerate(df.index))
    word_to_num = invert_dict(num_to_word)
    wv = df.as_matrix()
    return wv, word_to_num, num_to_word






##
# Utility functions used to create dataset
##
def augment_wv(df, extra=["UUUNKKK"]):
    for e in extra:
        df.loc[e] = zeros(len(df.columns))

def prune_wv(df, vocab, extra=["UUUNKKK"]):
    items = set(vocab).union(set(extra))
    return df.filter(items=items, axis='index')

def load_wv_raw(fname):
    return pd.read_table(fname, sep="\s+",
                         header=None,
                         index_col=0,
                         quoting=3)



def extract_tag_set(docs):
    tags = set(flatten1([[t[1].split("|")[0] for t in d] for d in docs]))
    return tags

def extract_word_set(docs):
    words = set(flatten1([[t[0] for t in d] for d in docs]))
    return words



##




def window_to_vec(window, L):
    return concatenate([L[i] for i in window])

##
# For fixed-window LM:
# each row of X is a list of word indices
# each entry of y is the word index to predict
def seq_to_lm_windows(words, word_to_num, ngram=2):
    ns = len(words)
    X = []
    y = []
    for i in range(ns):
        if words[i] == "<s>":
            continue # skip sentence begin, but do predict end
        idxs = [word_to_num[words[ii]]
                for ii in range(i - ngram + 1, i + 1)]
        X.append(idxs[:-1])
        y.append(idxs[-1])
    return array(X), array(y)

def docs_to_lm_windows(docs, word_to_num, ngram=2):
    docs = flatten1([pad_sequence(seq, left=(ngram-1), right=1)
                     for seq in docs])
    words = [canonicalize_word(wt[0], word_to_num) for wt in docs]
    return seq_to_lm_windows(words, word_to_num, ngram)


##
# For RNN LM
# just convert each sentence to a list of indices
# after padding each with <s> ... </s> tokens
def seq_to_indices(words, word_to_num):
    return array([word_to_num[w] for w in words])

def docs_to_indices(docs, word_to_num):
    docs = [pad_sequence(seq, left=1, right=1) for seq in docs]
    ret = []
    for seq in docs:
        words = [canonicalize_word(wt[0], word_to_num) for wt in seq]
        ret.append(seq_to_indices(words, word_to_num))

    # return as numpy array for fancier slicing
    return array(ret, dtype=object)

def offset_seq(seq):
    return seq[:-1], seq[1:]

def seqs_to_lmXY(seqs):
    X, Y = zip(*[offset_seq(s) for s in seqs])
    return array(X, dtype=object), array(Y, dtype=object)

##
# For RNN 
# return X, Y as lists
# where X[i] is indices, Y[i] is tags for a sequence
# NOTE: this does not use padding tokens!
#    (RNN should natively handle begin/end)
def docs_to_tag_sequence(docs, word_to_num, tag_to_num):
    # docs = [pad_sequence(seq, left=1, right=1) for seq in docs]
    X = []
    Y = []
    for seq in docs:
        if len(seq) < 1: continue
        words, tags = zip(*seq)

        words = [canonicalize_word(w, word_to_num) for w in words]
        x = seq_to_indices(words, word_to_num)
        X.append(x)

        tags = [t.split("|")[0] for t in tags]
        y = seq_to_indices(tags, tag_to_num)
        Y.append(y)

    # return as numpy array for fancier slicing
    return array(X, dtype=object), array(Y, dtype=object)

def idxs_to_matrix(idxs, L):

    return vstack([L[i] for i in idxs])