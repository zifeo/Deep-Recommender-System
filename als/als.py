# -*- coding: utf-8 -*-

import numpy as np
import scipy
from tqdm import tqdm
from random import sample

def rmse(data, user_features, item_features):
    nz_row, nz_col = data.nonzero()
    nz = list(zip(nz_row, nz_col))
    WZ = item_features.T @ user_features
    s = 0
    for u, i in nz:
        s += np.square(data[u, i] - WZ[u, i])
    return np.sqrt(s / len(nz))

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    num_item, num_user = train.shape
    item_features = np.random.random((num_features, num_item)) * np.sqrt(5 / num_features) # W
    user_features = np.random.random((num_features, num_user)) * np.sqrt(5 / num_features) # Z
    return user_features, item_features

def update_user_feature(ratings, user_features, item_features, lambda_user):
    """update user feature matrix."""
    num_item = ratings.shape[0]
    num_user = ratings.shape[1]
    num_features = item_features.shape[0]
    
    for i in tqdm(range(num_user), desc="update user"):
        nz = ratings[:, i].nonzero()[0]
        y = ratings[nz, i].todense()
        X = item_features[:, nz].T
        
        user_features.T[i] = np.squeeze(np.linalg.inv(X.T.dot(X) + lambda_user * np.eye(X.shape[1])).dot(X.T.dot(y)))
    return user_features

def update_item_feature(ratings, user_features, item_features, lambda_item):
    """update item feature matrix."""
    xs, ys = ratings.nonzero()
    
    num_item = ratings.shape[0]
    num_user = ratings.shape[1]
    num_features = user_features.shape[0]
    
    ratingsT = ratings.T
    
    for i in tqdm(range(num_item), desc="update item"):
        nz = ratingsT[:, i].nonzero()[0]
        y = ratingsT[nz, i].todense()
        X = user_features[:, nz].T
        
        item_features[:,i] = np.squeeze(np.linalg.inv(X.T.dot(X) + lambda_item * np.eye(X.shape[1])).dot(X.T.dot(y)))
    return item_features

def ALS(train, test, num_features, lambda_user, lambda_item, max_iter=1):
    """Alternating Least Squares (ALS) algorithm."""    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    tr_error = rmse(train, user_features, item_features)
    te_error = rmse(test, user_features, item_features)
    print("initial train rmse : ", tr_error, "\ninitial test rmse : ", te_error)

    i = 0
    while True:
        if i >= max_iter:
            break
            
        item_features = update_item_feature(train, user_features, item_features, lambda_item)
        user_features = update_user_feature(train, user_features, item_features, lambda_user)
        
        tr_error = rmse(train, user_features, item_features)
        te_error = rmse(test, user_features, item_features)
        print("train rmse : ", tr_error, "\ntest rmse : ", te_error)
        i += 1
         
    WZ = item_features.T @ user_features
    return WZ