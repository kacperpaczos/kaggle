import pandas as pd
from dt import DT

# Load the CSV file
import os

path = './Data'
file_name = 'titanic-homework.csv'
full_path = os.path.join(path, file_name)

if os.path.exists(full_path):
    print(f"File {file_name} exists.")
    df = pd.read_csv(full_path)

    # Remove the 'Name' column from DataFrame, some data are unnecessary.
    df = df.drop('Name', axis=1)

    ############

    dt = DT(df=df, column_to_predict='Survived', ignore_columns=['PassengerId'])
else:
    print(f"File {file_name} does not exist.")


