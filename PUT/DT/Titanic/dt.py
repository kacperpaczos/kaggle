from entropy import conditional_entropyXY, entropy
import pandas as pd
from icecream import ic

class Node:
    def __init__(self, df: pd.DataFrame, column_to_predict: str, ignore_columns: list[str], value=None, feature=None, leaf_label=None, node_label=None, is_leaf=False, unique_values=None, is_root=False):
        ic("##########################################")
        self.df = df
        self.feature = feature
        self.value = value
        self.column_to_predict = column_to_predict
        self.leaf_label = leaf_label
        self.node_label = node_label
        self.is_leaf = is_leaf
        self.is_root = is_root
        self.unique_values=unique_values

        if is_root:
            #ic("JESTEM ROOTEM")
            unique_values = {col: df[col].unique() for col in df.columns if col not in ignore_columns+[column_to_predict]}
            self.is_root=False
            self.unique_values = unique_values

        ic(unique_values)

        columns_list = df.columns.tolist()

        entropies = {}
        for col in columns_list:
            if col in [column_to_predict]:
                entropies[col] = entropy(y=df[col], base=2)

        conditional_entropies = {}
        for col in columns_list:
            if col not in ignore_columns+[column_to_predict]:
                conditional_entropies[col] = conditional_entropyXY(df[col], df[column_to_predict])

        information_gains = {col: entropies[column_to_predict] - conditional_entropies[col] for col in columns_list if col not in ignore_columns+[column_to_predict]}
        sorted_information_gains = sorted(information_gains.items(), key=lambda item: item[1], reverse=True)
        ic(sorted_information_gains)

        self.children = {}

        if not sorted_information_gains:
            self.is_leaf = True
            if not df[column_to_predict].mode().empty:
                self.value = df[column_to_predict].mode()[0]
                self.leaf_label = df[column_to_predict].mode()[0]
                ic(df[column_to_predict].mode()[0])
            else:
                self.value = '0'
                self.leaf_label = '0'
                ic("NNNNNNNNOONE")
            return

        best_feature = sorted_information_gains[0][0]
        ic(best_feature)
        self.feature = best_feature
        self.node_label = best_feature
        #ic(self.node_label)

        if entropies[column_to_predict] == 0:
            self.is_leaf = True
            import numpy as np
            unique_values, counts = np.unique(df[column_to_predict], return_counts=True)
            ic(unique_values)
            if len(unique_values) == 1:
                # Przypadek, gdy wszystkie próbki w węźle należą do tej samej klasy
                self.leaf_label = unique_values[0]
                ic("Jedyna wartość:", self.leaf_label)
                self.value = unique_values[0]
            elif len(counts) > 0:
                # Przypadek, gdy w węźle są próbki, ale nie wszystkie należą do tej samej klasy.
                # Wybieramy klasę, która pojawia się najczęściej.
                self.leaf_label = unique_values[np.argmax(counts)]
                ic("Najczęstsza wartość:", self.leaf_label)
                self.value = self.leaf_label
            else:
                # Przypadek, gdy nie ma próbek w węźle. Może się to zdarzyć, jeśli dane są niekompletne.
                # Tutaj można zdecydować, co zrobić w takim przypadku. Można na przykład przypisać domyślną wartość
                # lub wartość najczęściej występującą w całym zbiorze danych.
                # Na potrzeby przykładu przypisujemy wartość domyślną.
                self.leaf_label = "NIE"
                ic("Brak wartości:", self.leaf_label)
                self.value = "NIE"
            return
        

        for value in unique_values[best_feature]:
            filtered_df = df[df[best_feature] == value].drop(best_feature, axis=1)
            self.children[value] = Node(filtered_df, column_to_predict, ignore_columns, None, feature=best_feature, leaf_label=value, node_label=None, unique_values=self.unique_values, is_root=False)

    def is_leaf_node(self):
        return self.is_leaf

class DT:
    def __init__(self, df, column_to_predict, ignore_columns=[], is_root=True, unique_values=None):
        self.root = Node(df=df, column_to_predict=column_to_predict, ignore_columns=ignore_columns, is_root=is_root, unique_values=unique_values)

    def visualize_tree(self, node=None, prefix=""):
        if node is None:
            node = self.root
            tree_structure = {"NODE": "Root"}

        if node.is_leaf_node():
            leaf_name = f"Leaf: {node.leaf_label}, Value: {node.value}"
            tree_structure = {"LEAF": leaf_name}
        else:
            node_name = f"Node: {node.node_label}"
            if node.node_label is not None:
                node_name = f"{node.node_label}"  # Dodanie etykiety do nazwy węzła
            tree_structure = {"NODE": node_name, "children": []}
            for value, child in node.children.items():
                child_structure = self.visualize_tree(child, prefix + "    ")
                tree_structure["children"].append(child_structure)
                
        return tree_structure
