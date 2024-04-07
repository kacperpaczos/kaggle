from entropy import conditional_entropyXY, _entropy
import pandas as pd
import icecream as ic

class Node:
    def __init__(self, df: pd.DataFrame, column_to_predict: str, ignore_columns: list[str]):
        self.df = df
        # Zrób listę wszystkich kolumn z pliku CSV
        columns_list = df.columns.tolist()

        entropies = {}
        # entropia po prostu
        for col in columns_list:
            if col in [column_to_predict]:
                entropies[col] = _entropy(y=df[col], base=2)


        # entropia warunkowa kolumny
        conditional_entropies = {}
        for col in columns_list:
            if col not in ignore_columns:
                conditional_entropies[col] = conditional_entropyXY(df[col], df[column_to_predict])

        ic(conditional_entropies)

        information_gains = {col: entropies[column_to_predict] - conditional_entropies[col] for col in columns_list if col not in ignore_columns+[column_to_predict]}
        ic(information_gains)

        sorted_information_gains = sorted(information_gains.items(), key=lambda item: item[1], reverse=True)
        ic(sorted_information_gains)

        self.children={}

        unique_values = df[sorted_information_gains[0][0]].unique()
        for value in unique_values:
            filtered_df = df[df[sorted_information_gains[0][0]] == value].drop(sorted_information_gains[0][0], axis=1)
            self.children[value] = Node(filtered_df, column_to_predict, ignore_columns)

        # unique_values = list(set(df[sorted_information_gains[0][0]]))
        # for value in unique_values:
        #     self.children[value] = df[df[sorted_information_gains[0][0]] != value].drop(sorted_information_gains[0][0], axis=1)
            
        # list(set(df[sorted_information_gains[0][0]]))

        # dt.create_root(
        #     value = sorted_information_gains[0][1],
        #     children = children,
        #     feature = sorted_information_gains[0][0],
        #     threshold=None)
        # print(dt)

        df = df.drop('Name', axis=1)


        # self.feature = feature
        # self.threshold = threshold
        # self.children = children
        # self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DT:
    def __init__(self, df, column_to_predict, ignore_columns):
        self.root = Node(df=df, column_to_predict=column_to_predict, ignore_columns=ignore_columns)
        #self.df = df

#    def create_tree(self, df, column_to_predict):
#         children_dict = {key: Node(value=value) for key, value in children.items()}
        # self.root = Node(feature, threshold, children_dict, value=value)

