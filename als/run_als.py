#! /usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from helpers import load_data, split_data, predict
from als import ALS

path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)

num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)

pred = load_data("../data/sampleSubmission.csv")
nz = pred.nonzero()
WZ = ALS(train, test, 1, 0.9, 2.1)
WZ[WZ < 1] = 1
WZ[WZ > 5] = 5
pred[nz] = WZ[nz]
predict(pred)