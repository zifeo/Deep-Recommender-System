# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp


def read_txt(path):
    """Read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """Preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        
        # Inversed row and col !
        # Because ex10 data is inverd
        col, row = pos.split("_")
        row = row.replace("c", "")
        col = col.replace("r", "")
        
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings

def predict(predictions):
    """Write the prediction in a csv file"""
    with open("pred.csv", "w") as f:
        f.write("Id,Prediction\n")
        xs, ys = predictions.nonzero()
        for i in np.arange(xs.size):
            f.write("r{0}_c{1},{2}\n".format(ys[i]+1, xs[i]+1, predictions[xs[i], ys[i]]))

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """Split the ratings to training set and validation set.
    """
    # set seed
    np.random.seed(998)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users]  
    
    xs, ys = valid_ratings.nonzero()
    indices = list(zip(xs, ys))
    np.random.shuffle(indices)
    
    cut = int(p_test * len(indices))
    train = valid_ratings.copy()
    xs, ys = zip(*indices)
    train[xs[:cut], ys[:cut]] = 0
    test = valid_ratings.copy()
    test[xs[cut:], ys[cut:]] = 0

    return valid_ratings, train, test
