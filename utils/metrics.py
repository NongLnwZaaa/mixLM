import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    pre = precision_score(y_test, pred)

    rec = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    cm = confusion_matrix(y_test, pred)

    return acc, pre, rec, f1, cm