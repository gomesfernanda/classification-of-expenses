import argparse
import logging
import sys

import numpy as np
import pandas as pd
from keras import models, layers, optimizers, regularizers


# Maximum length of the description
MAXLEN = 100


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="XLSX data source", required=True)
    parser.add_argument("-o", "--output", help="Path to the trained model", required=True)
    parser.add_argument("--rows-to-skip", default=10, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--validation", default=0.15, type=float)
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
    df = loadData(args.input, args.rows_to_skip)
    X1 = np.zeros((len(df), MAXLEN), dtype=np.uint8)
    X2 = np.zeros((len(df),), dtype=np.float32)
    Y = np.zeros((len(df), df["Classification"].unique().size), dtype=np.int8)
    categories = {}
    for i, row in df.iterrows():
        desc = row["Concepto"]
        X1[i,MAXLEN - len(desc):] = [ord(c) for c in desc]
        X2[i] = row["Importe"]
        Y[i, categories.setdefault(row["Classification"], len(categories))] = 1
    X2 = (X2 - np.mean(X2)) / np.std(X2)

    model_desc = models.Sequential()
    embedding = np.zeros((256, 256), dtype=np.float32)
    np.fill_diagonal(embedding, 1)
    model_desc.add(layers.embeddings.Embedding(256, 256, input_length=MAXLEN, weights=[embedding], trainable=False))
    model_desc.add(layers.LSTM(128))

    model_amount = models.Sequential()
    model_amount.add(layers.Dense(10, input_shape=(1,), activation="relu"))

    merged = layers.Merge((model_desc, model_amount), mode="concat")
    final_model = models.Sequential()
    final_model.add(merged)
    final_model.add(layers.Dense(64, activation="relu"))
    final_model.add(layers.Dense(Y.shape[-1], activation="softmax"))

    final_model.compile(loss="categorical_crossentropy", optimizer="rmsprop",
                        metrics=["accuracy"], activity_regularizer=regularizers.l1(0.2))
    final_model.fit([X1, X2], Y, batch_size=50, epochs=args.epochs, validation_split=args.validation)
    print(final_model.summary())
    final_model.save(args.output)


if __name__ == "__main__":
    sys.exit(main())