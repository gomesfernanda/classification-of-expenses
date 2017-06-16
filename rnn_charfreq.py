import argparse
import logging
import sys

import numpy as np
import pandas as pd
from keras import models, layers, optimizers, regularizers
from keras.layers import Dense
import keras
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="XLSX data source", required=True)
    parser.add_argument("-o", "--output", help="Path to the trained model", required=True)
    parser.add_argument("--rows-to-skip", default=10, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--validation", default=0.15, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return args

def loadData(fileName, rowsToSkip):
    dfFunc = pd.read_excel(fileName, skiprows=rowsToSkip, parse_cols="B,F,G,K", parse_dates=['Fecha Operación'])
    datescol = pd.Series(dfFunc['Fecha Operación'])
    datesActualDay = datescol.dt.day
    datesTotalDays = datescol.dt.daysinmonth
    datesRel = datesActualDay / datesTotalDays
    dfFunc['FechaRel'] = datesRel
    dfFunc = dfFunc.dropna(axis=0, how='any')
    return dfFunc

def turnColumns(df):
    listConcepto = df['Concepto'].values
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, analyzer='char', lowercase=True)
    X_freqChar = vectorizer.fit_transform(listConcepto).toarray()
    X_Chars = vectorizer.get_feature_names()
    for row in range(0, len(df)):
        for charac in range(0, len(X_freqChar[row])):
            df.set_value(row, X_Chars[charac], X_freqChar[row][charac])
    return df

def main():
    args = setup()
    # loading my dataset and preparing the features
    df = loadData(args.input, args.rows_to_skip)
    df = turnColumns(df)
    dataset = df.values
    X = dataset[:, 3:]
    X[:,0] = (X[:,0] - np.mean(X[:,0])) / np.std(X[:,0])

    # defining my categories and turning into one-hot encoding
    Y = df.loc[:, ['Classification']]
    Y = Y.values
    Y = Y.flatten()
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    Y_onehot = np_utils.to_categorical(encoded_Y)

    # RNN model
    model= models.Sequential()
    model.add(Dense(len(X[0]), input_dim=len(X[0]), activation='relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(60, activation='relu', bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(23, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    csv_logger = keras.callbacks.CSVLogger('training.log')
    model.fit(X, Y_onehot, batch_size=150, epochs=args.epochs, validation_split=args.validation, verbose=2, shuffle=True, callbacks=[csv_logger])
    print(model.summary())
    model.save(args.output)

if __name__ == "__main__":
    sys.exit(main())