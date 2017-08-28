# coding=utf-8

import argparse
import logging
import sys
import time
import os
from shutil import copyfile

import numpy as np
import pandas as pd
import keras
from keras import models, layers, optimizers, regularizers, callbacks
from sklearn.metrics import confusion_matrix
from keras.callbacks import History

import seaborn as sns
import matplotlib.pyplot as plt


# Maximum length of the description
MAXLEN = 100


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="XLSX data source", required=True)
    parser.add_argument("-v", "--validationfile", help="Path to the validation file", required=True)
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


def main():
    args = setup()
    momentnow = time.strftime("%Y%m%d_%H%M%S")
    os.mkdir(momentnow)
    dfTrain = loadData(args.input, args.rows_to_skip)
    X1 = np.zeros((len(dfTrain), MAXLEN), dtype=np.uint8)
    X2 = np.zeros((len(dfTrain),), dtype=np.float32)
    X3 = np.zeros((len(dfTrain),), dtype=np.float32)
    Y = np.zeros((len(dfTrain), dfTrain["Classification"].unique().size), dtype=np.int8)
    categories = {}
    for i, row in dfTrain.iterrows():
        desc = row["Concepto"]
        X1[i,MAXLEN - len(desc):] = [ord(c) for c in desc]
        X2[i] = row["Importe"]
        X3[i] = row["FechaRel"]
        Y[i, categories.setdefault(row["Classification"], len(categories))] = 1
    X2 = (X2 - np.mean(X2)) / np.std(X2)
    inv_categories = {v: k for k, v in categories.items()}

    # preparing my prediction set
    dfPred = loadData(args.validationfile, args.rows_to_skip)
    X1pred = np.zeros((len(dfPred), MAXLEN), dtype=np.uint8)
    X2pred = np.zeros((len(dfPred),), dtype=np.float32)
    X3pred = np.zeros((len(dfPred),), dtype=np.float32)
    for i, row in dfPred.iterrows():
        desc = row["Concepto"]
        X1pred[i,MAXLEN - len(desc):] = [ord(c) for c in desc]
        X2pred[i] = row["Importe"]
        X3pred[i] = row["FechaRel"]
    X2pred = (X2pred - np.mean(X2pred)) / np.std(X2pred)

    # creating my RNN model
    model_desc = models.Sequential()
    embedding = np.zeros((256, 256), dtype=np.float32)
    np.fill_diagonal(embedding, 1)
    model_desc.add(layers.embeddings.Embedding(256, 256, input_length=MAXLEN, weights=[embedding], trainable=False))
    model_desc.add(layers.LSTM(128))

    model_amount = models.Sequential()
    model_amount.add(layers.Dense(10, input_shape=(1,), activation="relu"))

    model_date = models.Sequential()
    model_date.add(layers.Dense(10, input_shape=(1,), activation="relu"))

    merged = layers.Merge((model_desc, model_amount, model_date), mode="concat")
    final_model = models.Sequential()
    final_model.add(merged)
    final_model.add(layers.Dense(64, activation="relu"))
    final_model.add(layers.Dropout(args.dropout))
    final_model.add(layers.Dense(Y.shape[-1], activation="softmax"))
    final_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    csv_logger = keras.callbacks.CSVLogger(momentnow + "/metrics_" + momentnow + ".csv")
    modelfit = final_model.fit([X1, X2, X3], Y, batch_size=50, epochs=args.epochs, validation_split=args.validation, shuffle=True, callbacks=[csv_logger])
    print(final_model.summary())
    print(modelfit.history.keys())
    final_model.save(momentnow + "/model_" + momentnow + ".h5")
    final_model.to_json()

    # plot ACCURACY for training and validation sets
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(modelfit.history['acc'])
    plt.plot(modelfit.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plot LOSS for training and validation sets
    plt.subplot(1, 2, 2)
    plt.plot(modelfit.history['loss'])
    plt.plot(modelfit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(momentnow + "/plotloss_" + momentnow)
    plt.show()

    YPred = final_model.predict_classes([X1pred, X2pred, X3pred], verbose=2)
    YTrue = [categories[x] for x in dfPred["Classification"]]
    print(YPred)
    for i in range(0, len(YPred)):
        print(inv_categories[YPred[i]])
    hit = 0
    for i in range(0, len(YPred)):
        if YPred[i] == YTrue[i]:
            hit +=1
    acc_rate = hit/len(YPred)
    print("accuracy on prediction set: {:.6}%".format(acc_rate*100))
    labels = [inv_categories[x] for x in inv_categories]
    confusionmatrix = confusion_matrix(YTrue, YPred)
    cm_norm = confusionmatrix.astype("float") / confusionmatrix.sum(axis=1)[:, np.newaxis]
    cm_norm = np.round(cm_norm,2)
    sns.set(font_scale=0.9)  # for label size
    plt.figure()
    plotconf = sns.heatmap(cm_norm, annot=True, annot_kws={"size": 6}, cbar=False)
    plotconf.figure.savefig(momentnow + "/confusionmatrix_" + momentnow)

    destinationfile = momentnow + "/code_" + momentnow + '.py'
    copyfile(__file__, destinationfile)

if __name__ == "__main__":
    sys.exit(main())