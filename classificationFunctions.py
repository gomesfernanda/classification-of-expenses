import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def prepareData(fileName, rowsToSkip):
    # 01 // load the dataset, normalize the dates, deal with missing data
    def loadData(fileName, rowsToSkip):
        dfFunc = pd.read_excel(fileName, skiprows=rowsToSkip, parse_cols="B,F,G,K", parse_dates=['Fecha Operación'])
        datescol = pd.Series(dfFunc['Fecha Operación'])
        datesActualDay = datescol.dt.day
        datesTotalDays = datescol.dt.daysinmonth
        datesRel = datesActualDay / datesTotalDays
        dfFunc['FechaRel'] = datesRel
        dfFunc = dfFunc.dropna(axis=0, how='any')
        return dfFunc

    # 02 // define my vocabulary
    def myVocab(df):
        def buildVocab(given_str):
            prevocab = []
            for charact in given_str:
                if charact in prevocab:
                    next
                else:
                    prevocab.append(charact)
            vocab = sorted(prevocab)
            return vocab
        vocabList = []
        for row in range(0, len(df)):
            vocabList += buildVocab(df['Concepto'][row])
        finalVocab = sorted(set(vocabList))
        return finalVocab

    # 03 // count the characters for each row
    def countChar(df, finalVocab):
        for row in range(0, len(df)):
            dictprovis = dict([(finalVocab[i], 0) for i in range(len(finalVocab))])
            # freqrow = []
            for charact in df['Concepto'][row]:
                dictprovis[charact] = dictprovis[charact] + 1
            freqrow = list(dictprovis.values())
            # df.set_value(row, 'CharFreq Column', freqrow)
            for letvoc in range(0, len(finalVocab)):
                df.set_value(row, letvoc, freqrow[letvoc])
        return df
    # 04 // define my features to serve as iput for my neural network
    def ourFeatures(df):
        dataset = df.values
        X = dataset[:, 4:]
        Y = df.loc[:, ['Classification']]
        Y = Y.values
        Y = Y.flatten()
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # turn those integers into one-hot classification
        Y_onehot = np_utils.to_categorical(encoded_Y)
        return X, Y_onehot

    # 05 // call all functions above to return my features and classification
    df = loadData(fileName, rowsToSkip)
    finalVocab = myVocab(df)
    df = countChar(df, finalVocab)
    X, Y_onehot = ourFeatures(df)
    return X, Y_onehot

# 05 // define baseline model
def runNN(num_epochs, num_batch_size, verbose_012, num_splits, shuffle_BOOL, X, Y_onehot):
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(91, input_dim=91, kernel_initializer='normal', activation='relu'))
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        model.add(Dense(60, kernel_initializer='normal', activation='relu'))
        model.add(Dense(23, kernel_initializer='normal', activation='sigmoid'))
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    estimator = KerasClassifier(build_fn=baseline_model, epochs=num_epochs, batch_size=num_batch_size, verbose=verbose_012)
    kfold = KFold(n_splits=num_splits, shuffle=shuffle_BOOL, random_state=seed)
    results = cross_val_score(estimator, X, Y_onehot, cv=kfold)