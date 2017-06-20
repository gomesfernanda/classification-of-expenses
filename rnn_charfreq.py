import argparse
import logging
import sys

import numpy as np
import pandas as pd
from keras import models, layers, optimizers, regularizers
from keras.layers import Dense
import keras
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="XLSX data source", required=True)
    parser.add_argument("-o", "--output", help="Path to the trained model", required=True)
    parser.add_argument("-p", "--predictfile", help="Path to the prediction file", required=True)
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

def turnColumns(dfTrain, dfPred):
    unique_chars = []
    dfTrain['Concepto'] = dfTrain['Concepto'].str.upper()
    dfPred['Concepto'] = dfPred['Concepto'].str.upper()

    for row in range(0, len(dfTrain['Concepto'])):
        try:
            unique_chars += set(dfTrain['Concepto'][row])
            unique_chars += set(dfPred['Concepto'][row])
        except:
            unique_chars += set(dfTrain['Concepto'][row])

    final = sorted(set(unique_chars))

    for row in range(0, len(dfTrain['Concepto'])):
        mydict = dict(zip(final, [0] * len(final)))
        for charact in dfTrain['Concepto'][row]:
            mydict[charact] += 1
        for charact in final:
            dfTrain.set_value(row, charact, mydict[charact])

    for row in range(0, len(dfPred['Concepto'])):
        mydict = dict(zip(final, [0] * len(final)))
        for charact in dfPred['Concepto'][row]:
            mydict[charact] += 1
        for charact in final:
            dfPred.set_value(row, charact, mydict[charact])
    return dfTrain, dfPred

def main():
    args = setup()

    # loading my dataset and preparing the features (training and prediction)
    dfTrain = loadData(args.input, args.rows_to_skip)
    dfPred = loadData(args.predictfile, args.rows_to_skip)
    dfTrain, dfPred = turnColumns(dfTrain, dfPred)

    dataset_train = dfTrain.values
    XTrain = dataset_train[:, 3:]
    XTrain[:,0] = (XTrain[:,0] - np.mean(XTrain[:,0])) / np.std(XTrain[:,0])

    # preparing my prediction set
    dataset_pred = dfPred.values
    Xpred = dataset_pred[:, 3:]
    Xpred[:, 0] = (Xpred[:, 0] - np.mean(Xpred[:, 0])) / np.std(Xpred[:, 0])

    # defining my categories and turning into one-hot encoding
    YTrain = np.zeros((len(dfTrain), dfTrain["Classification"].unique().size), dtype=np.int8)
    categories = {}
    for i, row in dfTrain.iterrows():
        YTrain[i, categories.setdefault(row["Classification"], len(categories))] = 1
    YTrue = [categories[x] for x in dfPred['Classification']]
    inv_categories = {v: k for k, v in categories.items()}

    # RNN model
    model = models.Sequential()
    model.add(Dense(len(XTrain[0]), input_dim=len(XTrain[0]), activation='relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(60, activation='relu', bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(23, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    csv_logger = keras.callbacks.CSVLogger('training.log')
    model.fit(XTrain, YTrain, batch_size=150, epochs=args.epochs, validation_split=args.validation, verbose=2, shuffle=True, callbacks=[csv_logger])
    print(model.summary())
    model.save(args.output)

    # Predicting classes in an unknown dataset
    YPred = model.predict_classes(Xpred, verbose=1)

    hit = 0
    total = 0
    for i in range(0, len(YPred)):
        total+=1
        if YPred[i] == YTrue[i]:
            hit +=1
    print("accuracy on prediction: ", hit/total)

    confmat = confusion_matrix(YTrue, YPred)
    print(confmat)

if __name__ == "__main__":
    sys.exit(main())