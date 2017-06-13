import argparse
import logging
import sys

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import metrics, svm

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="XLSX data source", required=True)
    parser.add_argument("-v", "--validation", help="XLSX validation file", required=True)
    parser.add_argument("-f", "--filee", default="teste.csv")
    parser.add_argument("-o" "--output", help="ath to the trained model", required=False)
    parser.add_argument("--rows-to-skip", default=10, type=int)
    parser.add_argument("--n-gram", default=4, type=int)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return args

def loadData(fileName, rowsToSkip):
    df_loaded = pd.read_excel(fileName, skiprows=rowsToSkip, encoding='utf-8')
    return df_loaded

def main():
    args = setup()
    df_train = loadData(args.input, args.rows_to_skip)
    df_val = loadData(args.validation, args.rows_to_skip)

    X_train = df_train['Concepto'].values
    X_val = df_val['Concepto'].values

    y_train_text = df_train['Classification'].values
    y_val_text = list(df_val['Classification'].values)

    Y = np.zeros((len(df_train), df_train['Classification'].unique().size), dtype=np.int8)
    categories = {}
    for i, row in df_train.iterrows():
        Y[i, categories.setdefault(row["Classification"], len(categories))] = 1

    Y_tuple = [tuple(n) for n in Y]
    mydict = dict(zip(Y_tuple, y_train_text))

    classifier = Pipeline([
        ('trigram_vectorizer', CountVectorizer(ngram_range=(args.n_gram, args.n_gram), min_df=1, analyzer='char_wb')),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])

    classifier.fit(X_train, Y)

    predicted = classifier.predict(X_val)

    Y_predicted = []
    for item in range(0, len(X_val)):
        try:
            predict_tuple = tuple(predicted[item])
            Y_predicted.append(mydict[predict_tuple])
        except:
            Y_predicted.append('N/A')

    df_truepred = pd.DataFrame(np.column_stack([y_val_text, Y_predicted]), columns=['True Class', 'Predicted Class'])
    print(df_truepred)
    print("===================================")
    print("Accuracy on validation set: {:.4}".format(metrics.accuracy_score(y_val_text, Y_predicted)))
    df_truepred.to_csv(args.filee, sep=',', encoding='utf-8')

if __name__ == "__main__":
    sys.exit(main())