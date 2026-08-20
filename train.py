import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

data = []
labels = []

DATA_PATH = "data"

for label in os.listdir(DATA_PATH):
    for file in os.listdir(os.path.join(DATA_PATH, label)):
        path = os.path.join(DATA_PATH, label, file)
        arr = np.load(path)
        data.append(arr)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Accuracy:", acc)

joblib.dump(model, "model.pkl")