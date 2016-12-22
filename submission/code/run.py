# coding: utf-8

# import default scientific libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# import pandas for easy data management, keras for deep learning and scikit for feature generation
import pandas as pd
import keras
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko

# load needed feature
items = np.load('data/items.npy')
items.shape
items2 = np.load('data/items2.npy')
items2.shape
items3 = np.load('data/items3.npy')
items3.shape
users = np.load('data/users.npy')
users.shape
users2 = np.load('data/users2.npy')
users2.shape
users3 = np.load('data/users3.npy')
users3.shape


# ensure tensorflow does not leak
keras.backend.clear_session()


# load net model
model1 = km.load_model('data/model1.h5')



# compute weighted mean between with rating confidence and rating
def weighted_mean(preds):
    ret = []
    for e, s in zip(preds, np.argsort(preds, axis=1)):
        # highest confidence rating (index of the sorted array)
        #               |
        #               v
        n1, n2, n3, n4, n5 = s
        val = (n5 * e[n5] + n4 * e[n4] + n3 * e[n3] + n2 * e[n2] + n1 * e[n1]) / (e[n1] + e[n2] + e[n3] + e[n4] + e[n5])
        ret.append(val + 1)
    return np.array(ret)


# load submission data
sub = pd.read_csv('data/sampleSubmission.csv')
pos = sub.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)
sub['User'] = pos[0].astype(np.int)
sub['Item'] = pos[1].astype(np.int)
sub.head()


# compute prediction through the two network
pred2 = weighted_mean(model1.predict([sub.Item, sub.User, users[sub.Item - 1], items[sub.User - 1], users2[sub.Item - 1], items2[sub.User - 1], users3[sub.Item - 1], items3[sub.User - 1]], batch_size=1024))


# store prediction and output csv formatted for submission
sub['Prediction'] = pred2
sub.head()
sub.to_csv('submission.csv', columns=['Id', 'Prediction'], index=False)
