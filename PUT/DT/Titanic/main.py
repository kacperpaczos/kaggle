import pandas as pd
#import matplotlib.pyplot as plt
from dt import DT


# Wczytaj plik CSV
import os

sciezka = './Dane'
nazwa_pliku = 'titanic-homework.csv'
pelna_sciezka = os.path.join(sciezka, nazwa_pliku)

if os.path.exists(pelna_sciezka):
    print(f"Plik {nazwa_pliku} istnieje.")
    df = pd.read_csv(pelna_sciezka)

    # Usuń kolumnę 'Name' z DataFrame, niektore dane sa nam zbedne.
    df = df.drop('Name', axis=1)

    ############

    dt = DT(df=df, column_to_predict='Survived', ignore_columns=['PassengerId'])
else:
    print(f"Plik {nazwa_pliku} nie istnieje.")


