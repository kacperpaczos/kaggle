from entropy import conditional_entropyXY, entropy
import pandas as pd
from icecream import ic

class Node:
    def __init__(self, df: pd.DataFrame, column_to_predict: str, ignore_columns: list[str], value=None, feature=None, label=None):
        ic("##########################################")
        ic("DataFrame:", df)
        ic("Kolumna do przewidzenia:", column_to_predict)
        ic("Ignorowane kolumny:", ignore_columns)
        self.df = df
        self.feature = feature
        self.value = value
        self.column_to_predict=column_to_predict
        self.label=label
        # Zrób listę wszystkich kolumn z pliku CSV
        columns_list = df.columns.tolist()
        ic(columns_list)

        entropies = {}
        # entropia po prostu
        for col in columns_list:
            if col in [column_to_predict]:
                entropies[col] = entropy(y=df[col], base=2)

        ic(entropies)


        # entropia warunkowa kolumny
        conditional_entropies = {}
        for col in columns_list:
            if col not in ignore_columns+[column_to_predict]:
                conditional_entropies[col] = conditional_entropyXY(df[col], df[column_to_predict])

        ic(conditional_entropies)

        information_gains = {col: entropies[column_to_predict] - conditional_entropies[col] for col in columns_list if col not in ignore_columns+[column_to_predict]}
        ic(information_gains)

        sorted_information_gains = sorted(information_gains.items(), key=lambda item: item[1], reverse=True)
        ic(sorted_information_gains)

        self.children={}

        if not sorted_information_gains:  # Sprawdza, czy lista jest pusta
            self.value = df[column_to_predict].mode()[0]  # Ustaw najczęściej występującą wartość jako przewidywaną
            return  # Zakończenie budowy drzewa, jeśli nie ma zysków informacji
        
        # Najlepsza cecha do podziału (z największym zyskiem informacji)
        best_feature = sorted_information_gains[0][0]
        self.feature = best_feature  # Ustawienie najlepszej cechy jako atrybutu węzła

        unique_values = df[sorted_information_gains[0][0]].unique()
        ic(unique_values)
        for value in unique_values:
            #ic(value)
            filtered_df = df[df[sorted_information_gains[0][0]] == value].drop(sorted_information_gains[0][0], axis=1)
            self.children[value] = Node(filtered_df, column_to_predict, ignore_columns, None, feature=best_feature, label=value)

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

        #df = df.drop('Name', axis=1)


        # self.feature = feature
        # self.threshold = threshold
        # self.children = children
        # self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DT:
    def __init__(self, df, column_to_predict, ignore_columns):
        self.root = Node(df=df, column_to_predict=column_to_predict, ignore_columns=ignore_columns)

    def visualize_tree(self, node=None, indent="", tree_dict=None):
        if node is None:
            node = self.root
            tree_dict = {"name": "Root"}

        if node.is_leaf_node():
            leaf_name = f"Leaf Node: {node.column_to_predict} Predict = {node.value}"
            if node.label is not None:
                leaf_name = f"{node.label} -> {leaf_name}"  # Dodajemy etykietę do nazwy liścia
            tree_dict["name"] = leaf_name
        else:
            node_name = f"Node: {node.feature}"
            if node.label is not None:
                node_name = f"{node.label} -> {node_name}"  # Dodajemy etykietę do nazwy węzła
            tree_dict["name"] = node_name
            tree_dict["children"] = []
            for value, child in node.children.items():
                child_dict = {}
                tree_dict["children"].append(child_dict)
                self.visualize_tree(child, indent + "    ", child_dict)
                
        return tree_dict
                
        return tree_dict

    def printToPDFTree(tree_dict, filename="tree_visualization.pdf"):
        from graphviz import Digraph
        dot = Digraph(comment='Drzewo Decyzyjne')

        def dodaj_wezly_i_krawedzie(tree_dict, parent=None):
            if "children" in tree_dict:
                for child in tree_dict["children"]:
                    dot.node(child["name"], child["name"])
                    if parent:
                        dot.edge(parent, child["name"], constraint='false')
                    dodaj_wezly_i_krawedzie(child, child["name"])
            else:
                dot.node(tree_dict["name"], tree_dict["name"])
                if parent:
                    dot.edge(parent, tree_dict["name"], constraint='false')

        dodaj_wezly_i_krawedzie(tree_dict)
        dot.render(filename, view=True)
    
