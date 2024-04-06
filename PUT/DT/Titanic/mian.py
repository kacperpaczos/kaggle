import pandas as pd
import matplotlib.pyplot as plt
from PUT.DT.Titanic.dt import DT


# Wczytaj plik CSV
df = pd.read_csv('./Dane do zadań-20240303/titanic-homework.csv')

# Usuń kolumnę 'Name' z DataFrame, niektore dane sa nam zbedne.
df = df.drop('Name', axis=1)

############

dt = DT(df=df, column_to_predict='Survived', ignore_columns=['PassengerId'])

