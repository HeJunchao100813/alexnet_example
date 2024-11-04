import sys
assert sys.version_info >= (3, 7), "This notebook requires Python 3.7+"

from collections import Counter
import doctest

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

def read_data(filename):
    X = []
    y = []
    for line in open(filename).readlines():
        xi, yi = line.strip().split(" ")
        X.append(xi)
        y.append(yi)
    return train_test_split(X, y, test_size=0.5, random_state=42)

def count_ngrams(x, n):
    '''
The input can be a DNA string:
>>> count_ngrams("gactacgacacag", 3)
Counter({'gac': 2, 'aca': 2, 'act': 1, 'cta': 1, 'tac': 1,
         'acg': 1, 'cga': 1, 'cac': 1, 'cag': 1})

If shorter than n, there will be no n-grams:
>>> count_ngrams("gactacgacacag", 17)
Counter()

It can even be empty:
>>> count_ngrams("", 3)
Counter()

It can even be an ordinary (non-DNA) string:
>>> count_ngrams("abcdabcdefacd", 3)
Counter({'abc': 2, 'bcd': 2, 'cda': 1, 'dab': 1, 'cde': 1,
         'def': 1, 'efa': 1, 'fac': 1, 'acd': 1})
    '''
    # YOUR CODE HERE. MY CODE IS 1 LINE.
    return Counter(x[i:i+n] for i in range(len(x)-n+1))
    pass

def count_ngrams_multi(X, n):
    '''
>>> count_ngrams_multi(["catgcatg",
...                     "acacaca",
...                     "gtaaagtaaa"], 3)
[Counter({'cat': 2, 'atg': 2, 'tgc': 1, 'gca': 1}),
 Counter({'aca': 3, 'cac': 2}),
 Counter({'gta': 2, 'taa': 2, 'aaa': 2, 'aag': 1, 'agt': 1})]
>>> count_ngrams_multi([[]], 3)
[Counter()]
    '''
    # YOUR CODE HERE. MY CODE IS 1 LINE.
    pass


def counter_total(c):
    '''
    Sum of the counts in a counter c

    >>> counter_total(Counter())
    0
    >>> counter_total(Counter({'a': 1, 'b': 3}))
    4
    '''
    # YOUR CODE HERE. MY CODE IS 1 LINE.
    pass


def JD(A, B):
    '''Jaccard dissimilarity on multisets, represented by Counters.
    A multiset is like a set, as it's composed of unordered objects,
    but every item can occur more than once. It counts *how many times*
    each item occurs.

    Two are identical:
    >>> JD(Counter({'cat': 2, 'atg': 2, 'tgc': 1, 'gca': 1}),
    ...    Counter({'cat': 2, 'atg': 2, 'tgc': 1, 'gca': 1}))
    0.0

    Two are similar but not identical:
    >>> JD(Counter({'cat': 1, 'atg': 1, 'tgc': 1, 'gca': 1}),
    ...    Counter({'cat': 1, 'atg': 1, 'tgc': 1, 'gcg': 1}))
    0.4

    Worked example: in the above two Counters, their intersection is size 3 and their union is size 5.
    Therefore, the JD is 1 - 3/5 = 0.4.

    Notice that defining JD over sets would not be correct. The
    following Counters are the same if viewed as sets (they both have
    the same items) so JD would equal 0. But the correct implementation
    views them as multisets, ie it takes account of the number of each item.
    >>> JD(Counter({'cat': 2, 'atg': 1}),
    ...    Counter({'cat': 1, 'atg': 2}))
    0.5

    Two are totally different:
    >>> JD(Counter({'cat': 2, 'atg': 2, 'tgc': 1, 'gca': 1}),
    ...    Counter({'tca': 2, 'atc': 2, 'tga': 1, 'gct': 1}))
    1.0

    Note, the JD code doesn't have to work only on n-gram Counters. The same code should work on any Counters:
    >>> JD(Counter([0, 0, 0, 1, 1, 2, 3]),
    ...    Counter([0, 0, 0, 1, 1, 2, 4]))
    0.25

    '''
    # YOUR CODE HERE. MY CODE IS 3 LINES.
    pass

def distance_matrix(XA, XB, metric):
    """
    >>> XA = [Counter([1,1,1,2]), Counter([1,1,2,3])]
    >>> XB = [Counter([1,1,1,2]), Counter([1,1,2,4]), Counter([1,2,4])]
    >>> distance_matrix(XA, XB, JD)
    array([[0. , 0.4, 0.6],
           [0.4, 0.4, 0.6]])

    Location (0, 0) gives value 0.0 because XA[0] is equal to XB[0].
    Location (1, 2) gives value 0.6 because XA[1] and XB[2] have intersection of size 2 and union of size 5. JD gives 1 - 2/5 = 0.6.
    """
    # YOUR CODE HERE. MY CODE IS 5 LINES.
    result = np.zeros((len(XA), len(XB)))
    for i in range(len(XA)):
        for j in range(len(XB)):
            result[i, j] = metric(XA(i), XB(j))
    return result
    pass


def find_k_min_indices(X, k):
    """
    >>> X = np.array([[0.57, 0.53, 0.03, 0.19, 0.71],
    ...               [0.51, 0.91, 0.41, 0.27, 0.02],
    ...               [0.30, 0.37, 0.38, 0.57, 0.39],
    ...               [0.05, 0.37, 0.38, 0.83, 0.13]])
    >>> find_k_min_indices(X, 3)
    array([[2, 3, 1],
           [4, 3, 2],
           [0, 1, 2],
           [0, 4, 1]])
    >>> find_k_min_indices(X, 1) # k=1 makes sense
    array([[2],
           [4],
           [0],
           [0]])
    >>> find_k_min_indices(X, 10) # k=10 makes no sense as there are only 5 items per row
    Traceback (most recent call last):
    ...
    ValueError
    """

    # THIS IS COMPLETE: YOU DON'T NEED TO WRITE ANYTHING

    if k > X.shape[1]: raise ValueError

    # Get the indices that would sort each row of X
    sorted_indices = np.argsort(X, axis=1)

    # Return the first k indices for each row
    return sorted_indices[:, :k]

def vote(x):
    # easy way to vote. ties are broken arbitrarily.
    # eg Counter('aaaabbbccd').most_common(1) -> (('a', 4),)
    # so [0][0] -> 'a'
    # THIS FUNCTION IS COMPLETE. YOU DON'T NEED TO WRITE ANYTHING.
    return Counter(x).most_common(1)[0][0]

class NGramsKNN(BaseEstimator, ClassifierMixin):
    """
    A class representing k-nearest neighbours classification, using Jaccard dissimilarity
    as a measure of distance ("nearest") on counters of n-grams
    extracted from the sequences X.

    Doctests in the next cell.
    """
    # YOUR CODE HERE. MY CODE IS APPROX 25 LINES.


def test_NGramsKNN():
    """
    There should be a parameter n with default value as follows:
    >>> NGramsKNN().n
    3

    There should be a parameter k with default value as follows:
    >>> NGramsKNN().k
    3

    We can create the object, and if we don't over-ride default
    parameters, no parameter values are shown. Hint: in __init__, use super().
    >>> NGramsKNN()
    NGramsKNN()

    We can over-ride the default values, and then they are printed, eg:
    >>> NGramsKNN(k=5, n=4)
    NGramsKNN(k=5, n=4)

    Don't forget, fit() should return self:
    >>> X = ['abcdef', 'abcde', 'fedcbaaba', 'abcdefg', 'bbcde', 'ffdccba']
    >>> y = [1, 0, 1, 0, 0, 1]
    >>> NGramsKNN(n=2, k=1).fit(X, y)
    NGramsKNN(k=1, n=2)
    >>> NGramsKNN(n=2, k=1).fit(X, y).predict(['abcdef', 'fedcba'])
    array([1, 1])
    >>> NGramsKNN(n=2, k=3).fit(X, y).predict(['abcdef', 'fedcba'])
    array([0, 1])


    Before fitting, the object should not have the trailing-underscore attribute classes_:
    >>> nk = NGramsKNN(n=2, k=1)
    >>> hasattr(nk, "classes_")
    False

    After fitting, the object should have the trailing-underscore attribute classes_:
    >>> nk = NGramsKNN(n=2, k=1).fit(X, y)
    >>> hasattr(nk, "classes_")
    True
    """
    # YOU DON'T NEED TO WRITE ANYTHING IN THIS FUNCTION.
    pass