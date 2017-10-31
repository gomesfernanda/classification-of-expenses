import argparse
import logging
import sys
import time
import os
from shutil import copyfile

import numpy as np
import pandas as pd
from keras import models, layers, optimizers, regularizers
from keras.layers import Dense, Dropout
import keras
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

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
    datescol = pd.Series(dfFunc["Fecha Operación"])
    datesActualDay = datescol.dt.day
    datesTotalDays = datescol.dt.daysinmonth
    datesRel = datesActualDay / datesTotalDays
    dfFunc["FechaRel"] = datesRel
    dfFunc = dfFunc.dropna(axis=0, how="any")
    return dfFunc

def turnColumns(dfTrain, dfPred):
    unique_chars = []
    dfTrain["Concepto"] = dfTrain["Concepto"].str.upper()
    dfPred["Concepto"] = dfPred["Concepto"].str.upper()

    for row in range(0, len(dfTrain["Concepto"])):
        try:
            unique_chars += set(dfTrain["Concepto"][row])
            unique_chars += set(dfPred["Concepto"][row])
        except:
            unique_chars += set(dfTrain["Concepto"][row])

    final = sorted(set(unique_chars))

    for row in range(0, len(dfTrain["Concepto"])):
        mydict = dict(zip(final, [0] * len(final)))
        for charact in dfTrain["Concepto"][row]:
            mydict[charact] += 1
        for charact in final:
            dfTrain.set_value(row, charact, mydict[charact])

    for row in range(0, len(dfPred["Concepto"])):
        mydict = dict(zip(final, [0] * len(final)))
        for charact in dfPred["Concepto"][row]:
            mydict[charact] += 1
        for charact in final:
            dfPred.set_value(row, charact, mydict[charact])
    return dfTrain, dfPred

def main():
    args = setup()
    momentnow = time.strftime("%Y%m%d_%H%M%S")
    os.mkdir(momentnow)

    # loading my dataset and preparing the features (training and prediction)
    dfTrain = loadData(args.input, args.rows_to_skip)
    dfPred = loadData(args.validationfile, args.rows_to_skip)
    dfTrain, dfPred = turnColumns(dfTrain, dfPred)

    dataset_train = dfTrain.values
    XTrain = dataset_train[:, 3:]

    # preparing my prediction set
    dataset_pred = dfPred.values
    Xpred = dataset_pred[:, 3:]
    Xpred[:, 0] = (Xpred[:, 0] - np.mean(Xpred[:, 0])) / np.std(Xpred[:, 0])

    # defining my categories and turning into one-hot encoding
    YTrain = np.zeros((len(dfTrain), dfTrain["Classification"].unique().size), dtype=np.int8)
    categories = {}
    for i, row in dfTrain.iterrows():
        YTrain[i, categories.setdefault(row["Classification"], len(categories))] = 1
    YTrue = [categories[x] for x in dfPred["Classification"]]
    inv_categories = {v: k for k, v in categories.items()}

    # RNN model
    model = models.Sequential()
    model.add(Dense(len(XTrain[0]), input_dim=len(XTrain[0]), activation='relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(60, activation="relu", bias_regularizer=regularizers.l1(0.01)))
    model.add(Dense(23, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    csv_logger = keras.callbacks.CSVLogger(momentnow + "/metrics_" + momentnow + ".csv")
    finalmodel = model.fit(XTrain, YTrain, batch_size=150, epochs=args.epochs, validation_split=args.validation, verbose=2, shuffle=True, callbacks=[csv_logger])
    model.save(momentnow + "/model_" + momentnow + ".h5")
    model.to_json()

    # plot ACCURACY for training and validation sets
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(finalmodel.history['acc'])
    plt.plot(finalmodel.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plot LOSS for training and validation sets
    plt.subplot(1, 2, 2)
    plt.plot(finalmodel.history['loss'])
    plt.plot(finalmodel.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(momentnow + "/plotloss_" + momentnow)
    plt.show()

    # predicting classes in an unknown dataset
    YPred = model.predict_classes(Xpred, verbose=2)
    for i in range(0, len(YPred)):
        print(inv_categories[YPred[i]])
    hit = 0
    for i in range(0, len(YPred)):
        if YPred[i] == YTrue[i]:
            hit +=1
    acc_rate = hit/len(YPred)
    print("accuracy on prediction set: {:.6}%".format(acc_rate*100))

    # creating the confusion matrix
    YTrue_text = [inv_categories[x] for x in YTrue]
    YPred_text = [inv_categories[x] for x in YPred]
    labels = [inv_categories[x] for x in inv_categories]

    confusionmatrix = confusion_matrix(YTrue_text, YPred_text, labels=labels)
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