import numpy
import pandas
import keras
import keras.models
import keras.layers
from sklearn import metrics, model_selection
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

def data():
    
    ratings = pandas.read_csv('data_train.csv', dtype={'Prediction': numpy.int})
    pos = ratings.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)
    ratings['User'] = pos[0]
    ratings['Item'] = pos[1]
    ratings = ratings.iloc[:100000]

    y = numpy.zeros([ratings.shape[0], 5])
    y[numpy.arange(ratings.shape[0]), ratings.Prediction - 1] = 1
    y.shape

    train_u, test_u, train_i, test_i, train_y, test_y = model_selection.train_test_split(ratings.User, ratings.Item, y, test_size=0.5)
    
    X_train = [train_i[:], train_u[:]]
    Y_train = train_y[:]
    X_test = [test_i[:], test_u[:]]
    Y_test = test_y[:]
    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test):

    input_i = keras.layers.Input(shape=[1])
    i = keras.layers.Embedding(1000 + 1, 64)(input_i)
    i = keras.layers.Flatten()(i)
    i = keras.layers.Dropout({{uniform(0, 1)}})(i)

    input_u = keras.layers.Input(shape=[1])
    u = keras.layers.Embedding(10000 + 1, 64)(input_u)
    u = keras.layers.Flatten()(u)
    u = keras.layers.Dropout({{uniform(0, 1)}})(u)

    nn = keras.layers.merge([i, u], mode='concat')
    nn = keras.layers.Dense({{choice([128, 256, 512, 1024])}})(nn)
    nn = keras.layers.Activation({{choice(['relu', 'sigmoid'])}})(nn)
    nn = keras.layers.Dropout({{uniform(0, 1)}})(nn)
    nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = keras.layers.Dense({{choice([128, 256, 512, 1024])}})(nn)
    nn = keras.layers.Activation({{choice(['relu', 'sigmoid'])}})(nn)
    
    if conditional({{choice(['2', '3'])}}) == '3':
        nn = keras.layers.Dropout({{uniform(0, 1)}})(nn)
        nn = keras.layers.normalization.BatchNormalization()(nn)
        nn = keras.layers.Dense({{choice([128, 256, 512, 1024])}})(nn)
        nn = keras.layers.Activation({{choice(['relu', 'sigmoid'])}})(nn)

    output = keras.layers.Dense(5, activation='softmax')(nn)

    model = keras.models.Model([input_i, input_u], output)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
    
    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              validation_data=(X_test, Y_test))
    
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print(best_run)




