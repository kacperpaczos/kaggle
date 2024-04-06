import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# Wczytaj plik CSV
df = pd.read_csv('./Dane do zadań-20240303/titanic-homework.csv')

# Przygotowanie danych
# Usuń wiersze z brakującymi wartościami
df = df.dropna()

# Usuń wiersze z wartościami nieprawidłowymi
df = df[~df.isin([np.nan, np.inf, -np.inf]).any()]

# Podziel dane na cechy (X) i etykiety (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Podziel dane na zbiór treningowy i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja modelu
model = DecisionTreeClassifier(min_samples_split=2, max_depth=100)

# Trenowanie modelu
model.fit(X_train, y_train)

# Ocena modelu
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Dokładność modelu: ", accuracy)

# Wizualizacja drzewa
plt.figure(figsize=(20,10))
plot_tree(model, filled=True)
plt.show()
